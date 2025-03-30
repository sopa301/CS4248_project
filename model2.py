import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, AutoImageProcessor, AutoModelForImageClassification
import timm

class SelfAttention(nn.Module):
    def __init__(self, d_model=768, nhead=8, num_layers=2):
        super(SelfAttention, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, fused_tokens):
        fused_tokens = fused_tokens.transpose(0, 1)  # [seq_len, batch, d_model]
        fused = self.encoder(fused_tokens)
        return fused.transpose(0, 1)  # [batch, seq_len, d_model]


class SetImageEncoder(nn.Module):
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super(SetImageEncoder, self).__init__()
        self.device = device

        # Use timm to get Swin-B pretrained on ImageNet-22K
        self.swin = timm.create_model(
            'swin_base_patch4_window12_384_in22k',
            pretrained=True,
            features_only=False,
            num_classes=0  # removes classification head
        )

        self.image_proj = nn.Linear(1024, 768)
        encoder_layer = nn.TransformerEncoderLayer(d_model=768, nhead=8)
        self.fusion_transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)

    def encode_images(self, image_tensor_batch):
        image_tensor_batch = image_tensor_batch.to(self.device)
        with torch.no_grad():
            assert image_tensor_batch.ndim == 4, f"Expected 4D tensor, got {image_tensor_batch.shape}"
            features = self.swin.forward_features(image_tensor_batch)  # [B, 12, 12, 1024]
            features = features.view(features.size(0), -1, features.size(-1))  # [B, 144, 1024]
            pooled = features.mean(dim=1)                              # [B, 1024]
            # print(f"Image features shape: {features.shape}")
        return self.image_proj(pooled)                                 # [B, 768]


    def forward(self, batch_of_tensor_lists):
        fused_outputs = []
        for image_list in batch_of_tensor_lists:
            # image_list is a list of tensors [3, H, W]
            image_tensor_batch = torch.cat(image_list)  # [num_images, 3, H, W]
            # Feed the batch of images into Swin
            img_tokens = self.encode_images(image_tensor_batch).unsqueeze(1)  # [num_images, 1, 768]
            # print(f"Image tokens shape: {img_tokens.shape}")
            # Fuse across images
            fused = self.fusion_transformer(img_tokens)  # [num_images, 1, 768]
            pooled = fused.mean(dim=0)                   # [1, 768]
            fused_outputs.append(pooled.squeeze(0))      # [768]
            # print(f"Fused output shape: {fused_outputs[-1].shape}")

        return torch.stack(fused_outputs)                # [batch_size, 768]



class EmojiEncoder(nn.Module):
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super(EmojiEncoder, self).__init__()
        self.device = device
        self.bertweet = AutoModel.from_pretrained("vinai/bertweet-base").to(device)
        self.image_encoder = SetImageEncoder(device=device)
        self.self_attention = SelfAttention(d_model=768, nhead=8, num_layers=2)

    def forward(self, images, emoji_tokens):
        image_embeddings = self.image_encoder(images)  # [batch, 768]
        image_embeddings = image_embeddings.unsqueeze(1)  # [batch, 1, 768]
        encoded_emoji = self.bertweet(**emoji_tokens)
        emoji_embedding = encoded_emoji.last_hidden_state  # [batch, seq_len, 768]
        fused_tokens = torch.cat([emoji_embedding, image_embeddings], dim=1)  # [batch, seq+1, 768]
        return self.self_attention(fused_tokens)  # [batch, seq+1, 768]


class CrossModal(nn.Module):
    def __init__(self, d_model=768, nhead=8, num_layers=2):
        super(CrossModal, self).__init__()
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

    def forward(self, text_embeddings, fused_representation):
        text_embeddings = text_embeddings.transpose(0, 1)
        fused_representation = fused_representation.transpose(0, 1)
        cross_fused = self.decoder(tgt=text_embeddings, memory=fused_representation)
        return cross_fused.transpose(0, 1)


class FinalModel(nn.Module):
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super(FinalModel, self).__init__()
        self.device = device
        self.fusion_model = EmojiEncoder(device=device)
        self.eng_encoder = AutoModel.from_pretrained("bert-base-uncased").to(device)
        self.cross_modal = CrossModal(d_model=768, nhead=8, num_layers=2)
        self.classifier = nn.Linear(768, 2)

    def forward(self, images, emoji_tokens, text_tokens):
        fused_output = self.fusion_model(images, emoji_tokens)
        encoded_text = self.eng_encoder(**text_tokens)
        text_embeddings = encoded_text.last_hidden_state
        cross_fused = self.cross_modal(text_embeddings, fused_output)
        aggregated = cross_fused.mean(dim=1)
        logits = self.classifier(aggregated)
        return logits