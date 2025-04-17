import os
import torch
import torch.nn as nn
from transformers import AutoModel
from torchvision.models import swin_transformer

fusion_hidden_size = 768
class MultiImageEncoder(nn.Module):
    def __init__(self, swin_model, output_dim=768, max_num_images=10):  # Add max_num_images
        super().__init__()
        self.swin = swin_model
        self.image_proj = nn.Linear(swin_model.norm.normalized_shape[0], output_dim)

        self.positional_embedding = nn.Parameter(torch.randn(max_num_images, output_dim))  # New line

        encoder_layer = nn.TransformerEncoderLayer(d_model=output_dim, nhead=8)
        self.fusion_transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)

        for param in self.swin.parameters():
            param.requires_grad = False


    def forward(self, batch_of_image_lists):
        batch_embeddings = []
        max_seq_len = self.positional_embedding.size(0)

        for images in batch_of_image_lists:
            image_tensor = torch.stack(images).to(next(self.swin.parameters()).device)
            with torch.no_grad():
                features = self.swin(image_tensor)

            projected = self.image_proj(features)  # Shape: (num_images, output_dim)
            
            seq_len = projected.size(0)
            if seq_len > max_seq_len:
                raise ValueError(f"Number of images ({seq_len}) exceeds max_num_images ({max_seq_len})")

            # Add positional embedding
            pos_embed = self.positional_embedding[:seq_len, :]  # Shape: (seq_len, output_dim)
            fused_input = projected + pos_embed

            # Transformer expects shape (seq_len, batch, dim)
            fused_input = fused_input.unsqueeze(1)
            fused_output = self.fusion_transformer(fused_input)

            pooled = fused_output.mean(dim=0).squeeze(0)
            batch_embeddings.append(pooled)

        return torch.stack(batch_embeddings)



class MultimodalFusion(nn.Module):
    def __init__(self, text_dim, img_dim, hidden_size=768):
        super().__init__()
        self.hidden_size = hidden_size

        self.image_projection = nn.Linear(img_dim, hidden_size)

        self.text_query = nn.Linear(hidden_size, hidden_size)
        self.img_key = nn.Linear(hidden_size, hidden_size)
        self.img_value = nn.Linear(hidden_size, hidden_size)

        self.img_query = nn.Linear(hidden_size, hidden_size)
        self.text_key = nn.Linear(hidden_size, hidden_size)
        self.text_value = nn.Linear(hidden_size, hidden_size)

        self.text_norm = nn.LayerNorm(hidden_size)
        self.img_norm = nn.LayerNorm(hidden_size)

        self.fusion = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU()
        )

        self.gate = nn.Sequential(
            nn.Linear(hidden_size * 2, 2),
            nn.Softmax(dim=-1)
        )

        self._init_weights()

    def _init_weights(self):
        for module in [self.text_query, self.img_key, self.img_value,
                       self.img_query, self.text_key, self.text_value,
                       self.image_projection]:
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            nn.init.zeros_(module.bias)

    def forward(self, text_features, image_features):
        batch_size = text_features.shape[0]
        cls_token = text_features[:, 0]

        img_projected = self.image_projection(image_features)

        img_q = self.img_query(img_projected).view(batch_size, 1, -1)
        text_k = self.text_key(text_features)
        text_v = self.text_value(text_features)

        img_attention_scores = torch.matmul(img_q, text_k.transpose(-1, -2)) / (self.hidden_size ** 0.5)
        img_attention_weights = torch.softmax(img_attention_scores, dim=-1)
        img_context = torch.matmul(img_attention_weights, text_v).squeeze(1)
        img_context = self.img_norm(img_context + img_projected)

        text_q = self.text_query(cls_token).view(batch_size, 1, -1)
        img_k = self.img_key(img_projected).view(batch_size, 1, -1)
        img_v = self.img_value(img_projected).view(batch_size, 1, -1)

        text_attention_scores = torch.matmul(text_q, img_k.transpose(-1, -2)) / (self.hidden_size ** 0.5)
        text_attention_weights = torch.softmax(text_attention_scores, dim=-1)
        text_context = torch.matmul(text_attention_weights, img_v).squeeze(1)
        text_context = self.text_norm(text_context + cls_token)

        combined = torch.cat([text_context, img_context], dim=1)
        gate_weights = self.gate(combined)

        text_weight = gate_weights[:, 0].unsqueeze(1)
        img_weight = gate_weights[:, 1].unsqueeze(1)

        weighted_text = text_context * text_weight
        weighted_img = img_context * img_weight

        multimodal_features = torch.cat([weighted_text, weighted_img], dim=1)
        output = self.fusion(multimodal_features)

        return output


class EmoteMultimodalModel(nn.Module):
    def __init__(self, config, num_labels=2):
        super().__init__()
        self.config = config
        self.num_labels = num_labels

        self.text_model = AutoModel.from_pretrained("textattack/bert-base-uncased-MNLI")

        swin_backbone = swin_transformer.swin_b(weights="DEFAULT")
        swin_backbone.head = nn.Identity()

        self.image_model = MultiImageEncoder(swin_backbone, output_dim=fusion_hidden_size)

        self.text_dim = 768
        self.img_dim = fusion_hidden_size

        self.fusion = MultimodalFusion(self.text_dim, self.img_dim, fusion_hidden_size)

        self.dropout = nn.Dropout()
        self.classifier = nn.Linear(fusion_hidden_size, num_labels)

        self._init_weights()

        self.gradient_clip_val = 1.0
        self.lr_warmup_steps = 100
        self.total_steps = 0

    def _init_weights(self):
        nn.init.normal_(self.classifier.weight, std=0.01)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, images=None, labels=None, **kwargs):
        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True
        )

        text_features = text_outputs.last_hidden_state
        image_features = self.image_model(images)

        fused_features = self.fusion(text_features, image_features)
        fused_features = self.dropout(fused_features)
        logits = self.classifier(fused_features)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)

        self.total_steps += 1

        return {"loss": loss, "logits": logits}
