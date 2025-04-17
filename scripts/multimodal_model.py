import os
import torch
import torch.nn as nn
from torchvision.models import swin_transformer

class EmoteMultimodalModel(nn.Module):
    def __init__(self, config, num_labels=2):
        super().__init__()
        self.config = config
        self.num_labels = num_labels
        
        # Load Swin but drastically simplify
        self.image_model = swin_transformer.swin_t(weights="DEFAULT")
        self.image_model.head = nn.Identity()
        
        # Freeze Swin
        for param in self.image_model.parameters():
            param.requires_grad = False
        
        # Aggressively reduce feature dimensions - emojis are simple
        self.projection = nn.Sequential(
            nn.Linear(768, 32),  # 24x reduction
            nn.ReLU()
        )
        
        # Option 1: Tiny GRU - extremely parameter efficient
        self.rnn = nn.GRU(
            input_size=32,
            hidden_size=32,
            batch_first=True
        )
        
        # Simple classifier
        self.classifier = nn.Linear(32, num_labels)
        
        # Initialize weights
        nn.init.normal_(self.projection[0].weight, std=0.02)
        nn.init.zeros_(self.projection[0].bias)
        nn.init.normal_(self.classifier.weight, std=0.02)
        nn.init.zeros_(self.classifier.bias)
        
    def forward(self, images=None, labels=None, **kwargs):
        batch_size, seq_len, c, h, w = images.shape
        
        # Process all images at once
        images_flat = images.view(batch_size * seq_len, c, h, w)
        
        # Extract features using frozen Swin
        with torch.no_grad():
            features = self.image_model(images_flat)
            
        # Reshape back to sequence
        features = features.view(batch_size, seq_len, -1)
        
        # Project to minimal dimension
        projected = self.projection(features)  # [B, seq_len, 32]
        
        # Process sequence
        sequence_out, _ = self.rnn(projected)
        final_state = sequence_out[:, -1, :]  # Take final state
        
        # Classification
        logits = self.classifier(final_state)
        
        # Loss calculation
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            
        return type('obj', (object,), {'loss': loss, 'logits': logits})
    
    def save_pretrained(self, save_directory):
        if os.path.isfile(save_directory): print(f"Path should be directory"); return
        os.makedirs(save_directory, exist_ok=True)
        
        model_config = {
            "image_model_name": self.config.image_model_name,
            "num_labels": self.num_labels,
            "img_feature_dim": 32,
            "architecture": "sequential_emoji_rnn"
        }
        
        # Save only the RNN and classifier parts
        torch.save(
            {
                "model_config": model_config,
                "rnn_state_dict": self.rnn.state_dict(),
                "classifier_state_dict": self.classifier.state_dict(),
            },
            os.path.join(save_directory, "sequential_emoji_model.pt")
        )
        return save_directory
    
    @classmethod
    def from_pretrained(cls, pretrained_model_path, config, **kwargs):
        components_path = os.path.join(pretrained_model_path, "sequential_emoji_model.pt")
        if not os.path.exists(components_path):
            raise ValueError(f"Could not find model at {components_path}")
            
        components = torch.load(components_path, map_location="cpu")
        model_config = components["model_config"]
        
        # Create a new model instance
        model = cls(config, num_labels=model_config.get("num_labels", 2))
        
        # Load the saved components
        model.rnn.load_state_dict(components["rnn_state_dict"])
        model.classifier.load_state_dict(components["classifier_state_dict"])
        
        return model 