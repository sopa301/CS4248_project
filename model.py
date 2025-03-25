import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from transformers import AutoTokenizer, AutoModel
from transformers import AutoImageProcessor, AutoModelForImageClassification
from torch.nn.utils.rnn import pad_sequence


class SelfAttention(nn.Module):
    """
    Fuse an emoji image and its Unicode representation.
    Uses self-attention over concatenated tokens.
    """
    def __init__(self, d_model=768, nhead=8, num_layers=1):
        super(SelfAttention, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
    
    def forward(self, fused_tokens):
        # fused_tokens: [batch, seq_len, d_model] (concatenated unicode and image token)
        # Transformer expects shape [seq_len, batch, d_model]
        fused_tokens = fused_tokens.transpose(0, 1)
        fused = self.encoder(fused_tokens)
        return fused.transpose(0, 1)

class EmojiEncoder(nn.Module):
    """
    Process the emoji image and Unicode text.
    - Image is processed with Swin Transformer.
    - Unicode text is processed with BERTweet.
    - Their tokens are concatenated and fused with self-attention.
    """
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super(EmojiEncoder, self).__init__()
        self.device = device
        
        self.processor = AutoImageProcessor.from_pretrained("microsoft/swinv2-base-patch4-window12-192-22k")
        self.swin = AutoModelForImageClassification.from_pretrained("microsoft/swinv2-base-patch4-window12-192-22k")
        # Remove the classification head.
        self.swin.head = nn.Identity()
        # Project image features to 768 dimensions.
        SWIN_FEATURES = 21841
        self.swin_proj = nn.Linear(SWIN_FEATURES, 768)
        
        # Unicode Text Encoder: BERTweet.
        self.unicode_tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base", normalization=True)
        self.bertweet = AutoModel.from_pretrained("vinai/bertweet-base")
        
        # Fusion with self-attention.
        self.self_attention = SelfAttention(d_model=768, nhead=8, num_layers=1)
        
    
    def forward(self, images: list[list], emojis: list):
        # --- Process the image ---
        batch_images = []
        batch_emojis = []
        for batch in images:
            image_tokens = []
            for image in batch:
                if isinstance(image, Image.Image):
                    processed = self.processor(image, return_tensors="pt").to(self.device)
                    image_tensor = processed["pixel_values"].to(self.device)
                else:
                    image_tensor = image.to(self.device)
                # Get image feature vector.
                swin_output = self.swin(image_tensor)            # [ swin_features]
                img_feats = swin_output.logits 
                img_feats = self.swin_proj(img_feats)            # [ 768]
                # Treat image feature as a single token.
                img_token = img_feats.unsqueeze(0)              # [ 1, 768]
                image_tokens.append(img_token)
            img_tokens = torch.cat(image_tokens, dim=0)          # [ num_images, 768]
            batch_images.append(img_tokens)

            # --- Process the Unicode text ---
            # Unicode text is expected to be a string (or a list for batching).
            emoji_tokens = []
            for emoji in emojis:
                encoded_input = self.unicode_tokenizer(emoji, return_tensors='pt', padding=True, truncation=True).to(self.device)
                emoji_output = self.bertweet(**encoded_input)
                emoji_token = emoji_output.last_hidden_state  # [num_unicode, seq_len, 768]
                #TODO: think of different way to concatenate the unicode tokens
                emoji_token = emoji_token.reshape(-1, 768)       # [seq_len, 768]
                emoji_tokens.append(emoji_token)

                
            emoji_tokens = torch.cat(emoji_tokens, dim=0)  # [ total_emoji_seq_len, 768]
            batch_emojis.append(emoji_tokens)
        all_emoji_tokens = pad_sequence(batch_emojis, batch_first=True)  # [batch, total_emoji_seq_len, 768]
        all_img_tokens = torch.squeeze(pad_sequence(batch_images, batch_first=True))  # [batch, num_images, 768]

        

        # --- Concatenate and Fuse ---
        # Concatenate along the sequence dimension.
        fused_tokens = torch.cat([all_emoji_tokens, all_img_tokens], dim=1)  # [batch, total_emoji_seq_len+num_images, 768]
        fused_output = self.self_attention(fused_tokens)       # [batch, total_emoji_seq_len+num_images, 768]
        
        return fused_output


class CrossModal(nn.Module):
    """
    Let the English text attend to the fused emoji image + Unicode representation.
    """
    def __init__(self, d_model=768, nhead=8, num_layers=1):
        super(CrossModal, self).__init__()
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
    
    def forward(self, eng_tokens, fused_rep):
        # eng_tokens: [batch, seq_len_eng, d_model]
        # fused_rep: [batch, seq_len_fused, d_model]
        # Transformer expects shape [seq_len, batch, d_model]
        eng_tokens = eng_tokens.transpose(0, 1)
        fused_rep = fused_rep.transpose(0, 1)
        # Let English text tokens (as queries) attend to the fused tokens (as memory).
        cross_fused = self.decoder(tgt=eng_tokens, memory=fused_rep)
        return cross_fused.transpose(0, 1)  # [batch, seq_len_eng, d_model]

class FinalModel(nn.Module):
    """
    Full model that:
      1. Fuses an emoji image with its Unicode representation (using self-attention).
      2. Encodes an English sentence.
      3. Uses cross-modal attention for the English text to attend to the fused representation.
      4. (Optionally) Aggregates the final features for classification.
    """
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super(FinalModel, self).__init__()
        self.device = device
        
        # Part 1: Fuse emoji image and unicode.
        self.fusion_model = EmojiEncoder(device=device)
        
        # English Text Encoder: Using a BERT-based model.
        self.eng_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.eng_encoder = AutoModel.from_pretrained("bert-base-uncased")
        
        # Cross-modal fusion: English text attends to the fused output.
        self.cross_modal = CrossModal(d_model=768, nhead=8, num_layers=1)
        
        # Optional classification head.
        self.classifier = nn.Linear(768, 2)  # For binary classification, adjust as needed.
    
    def forward(self, images: list[Image.Image], emojis: list[str], english_text: str):
        # 1. Fuse emoji image and unicode text.
        fused_output = self.fusion_model(images, emojis)   # [batch, seq_len_fused, 768]
        
        # 2. Encode the English sentence.
        encoded_eng = self.eng_tokenizer(english_text, return_tensors='pt', padding=True, truncation=True).to(self.device)
        eng_output = self.eng_encoder(**encoded_eng)
        eng_tokens = eng_output.last_hidden_state              # [batch, seq_len_eng, 768]
        
        # 3. Cross-modal fusion: let English tokens attend to fused emoji/unicode representation.
        cross_fused = self.cross_modal(eng_tokens, fused_output)  # [batch, seq_len_eng, 768]
        
        # 4. For classification, aggregate (e.g., via mean pooling) the English tokens.
        aggregated = cross_fused.mean(dim=1)                     # [batch, 768]
        logits = self.classifier(aggregated)                     # [batch, num_classes]
        return logits


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = FinalModel(device=device).to(device)
    
    # Load an emoji image (update the path accordingly).
    image_path = "/home/andrew/CS4248_project/dataset/noto-emoji/png/512/emoji_u1f923.png"
    image = Image.open(image_path).convert("RGB")
    
    # Unicode text representing the emoji (this might be a short string, e.g., "🤣").
    emojis = "🤣"
    
    # An English sentence for the entailment task.
    english_text = "This emoji expresses happiness."
    
    # Forward pass.
    logits = model(image, emojis, english_text)
    print("Logits:", logits)
