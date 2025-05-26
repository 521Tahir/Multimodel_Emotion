#!/usr/bin/env python
# coding: utf-8

# In[57]:


# ===========================================
# 1. IMPORTS & PATHS
# ===========================================
import matplotlib.pyplot as plt
import torch.nn.functional as F
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer, BertModel

# === Dataset Paths ===
CMU_PATH = r"D:/New folder/data/CMU-MOSEI"
CMU_CSV = r"D:/New folder/data.csv"
IEMOCAP_PATH = r"C:/Users/hp/Downloads/archive"
IEMOCAP_CSV = os.path.join(IEMOCAP_PATH, "data.csv")


def find_audio_subfolder(filename, audio_base):
    for subfolder in ['Train_modified', 'Val_modified', 'Test_modified']:
        full_path = os.path.join(audio_base, subfolder, filename)
        if os.path.exists(full_path):
            return subfolder
    return None

print("Loading CMU-MOSEI and IEMOCAP CSVs...")
df_cmu = pd.read_csv(CMU_CSV)
df_cmu['dataset'] = 'cmu'
if 'audio_file' in df_cmu.columns:
    df_cmu['audio_subfolder'] = df_cmu['audio_file'].apply(lambda f: find_audio_subfolder(f, os.path.join(CMU_PATH, 'Audio Chunk')))

if os.path.exists(IEMOCAP_CSV):
    df_iemocap = pd.read_csv(IEMOCAP_CSV)
    df_iemocap['dataset'] = 'iemocap'
    if 'audio_file' in df_iemocap.columns:
        df_iemocap['audio_subfolder'] = df_iemocap['audio_file'].apply(lambda f: find_audio_subfolder(f, os.path.join(IEMOCAP_PATH, 'Audio Chunk')))
    data_df = pd.concat([df_cmu, df_iemocap], ignore_index=True)
else:
    print("⚠️ IEMOCAP CSV not found, using CMU-MOSEI only")
    data_df = df_cmu

print("Combined data preview:")
print(data_df.head())


# In[59]:


# ===========================================
# 2. FEATURE EXTRACTORS & DATASET WRAPPER
# ===========================================
class DummyAudioExtractor:
    def __call__(self, path):
        return torch.randn(100, 300)

class DummyVisualExtractor:
    def __call__(self, path):
        return torch.randn(100, 300)

class TextExtractor:
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert = BertModel.from_pretrained('bert-base-uncased')

    def __call__(self, text):
        tokens = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True)
        with torch.no_grad():
            outputs = self.bert(**tokens)
        return outputs.last_hidden_state.squeeze(0)


class MultimodalDataset(Dataset):
    def __init__(self, dataframe, audio_extractor, visual_extractor, text_extractor):
        self.dataframe = dataframe.reset_index(drop=True)
        self.audio_extractor = audio_extractor
        self.visual_extractor = visual_extractor
        self.text_extractor = text_extractor
        label_column = 'label' if 'label' in dataframe.columns else 'emotion'
        self.label_encoder = LabelEncoder()
        self.labels = self.label_encoder.fit_transform(dataframe[label_column])

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        dataset = row['dataset']
        base_path = CMU_PATH if dataset == 'cmu' else IEMOCAP_PATH

        audio_path = None
        if pd.notna(row.get('audio_file')) and pd.notna(row.get('audio_subfolder')):
            audio_path = os.path.join(base_path, 'Audio Chunk', row['audio_subfolder'], str(row['audio_file']))

        video_path = os.path.join(base_path, 'Visual Chunk', str(row['video_file'])) if pd.notna(row.get('video_file')) else None
        text_data = row['text'] if pd.notna(row.get('text')) else ""

        audio = self.audio_extractor(audio_path) if audio_path else torch.zeros(100, 300)
        visual = self.visual_extractor(video_path) if video_path else torch.zeros(100, 300)
        text = self.text_extractor(text_data) if text_data else torch.zeros(100, 768)

        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return audio, visual, text, label


# In[61]:


# 3. MODEL ARCHITECTURE
# ===========================================
class ModalityTransformer(nn.Module):
    def __init__(self, input_dim=300, hidden_dim=128, nhead=4):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=nhead, dim_feedforward=hidden_dim)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)

    def forward(self, x):
        return self.transformer(x)


class MultimodalFusion(nn.Module):
    def __init__(self, audio_dim=300, visual_dim=300, text_dim=768):
        super().__init__()
        total_dim = audio_dim + visual_dim + text_dim
        self.fc = nn.Sequential(
            nn.Linear(total_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 6)
        )

    def forward(self, a, v, t):
        avg_a = a.mean(dim=1)
        avg_v = v.mean(dim=1)
        avg_t = t.mean(dim=1)
        x = torch.cat([avg_a, avg_v, avg_t], dim=-1)
        return self.fc(x)


class EmotionClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.audio_trans = ModalityTransformer()
        self.visual_trans = ModalityTransformer()
        self.text_trans = ModalityTransformer(input_dim=768)
        self.fusion = MultimodalFusion(audio_dim=300, visual_dim=300, text_dim=768)

    def forward(self, a, v, t):
        a = self.audio_trans(a)
        v = self.visual_trans(v)
        t = self.text_trans(t)
        return self.fusion(a, v, t)


# In[63]:


# 4. CROSS-MODAL TRANSLATION MODULES
# ===========================================
class GeneralTranslator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
        self.decoder = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )

    def forward(self, x):
        latent = self.encoder(x)
        recon = self.decoder(latent)
        return recon, latent


class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)


class CrossModalTranslator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
        self.decoder = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )

    def forward(self, x):
        latent = self.encoder(x)
        recon = self.decoder(latent)
        return recon, latent


def kl_divergence(mu):
    return -0.5 * torch.sum(1 + 0 - mu.pow(2) - 0, dim=1).mean()


def sequence_discrepancy(real_seq, fake_seq):
    return F.mse_loss(real_seq.unsqueeze(1), fake_seq.unsqueeze(1))


def train_cross_modal_gan(text_feats, audio_feats, translator, discriminator, optim_trans, optim_disc, recon_crit, adv_crit):
    discriminator.train()
    real_labels = torch.ones(audio_feats.size(0), 1)
    fake_labels = torch.zeros(audio_feats.size(0), 1)

    fake_audio, _ = translator(text_feats)
    pred_real = discriminator(audio_feats.detach())
    pred_fake = discriminator(fake_audio.detach())

    loss_real = adv_crit(pred_real, real_labels)
    loss_fake = adv_crit(pred_fake, fake_labels)
    loss_disc = (loss_real + loss_fake) / 2

    optim_disc.zero_grad()
    loss_disc.backward()
    optim_disc.step()

    translator.train()
    fake_audio, _ = translator(text_feats)
    pred_fake = discriminator(fake_audio)
    loss_adv = adv_crit(pred_fake, real_labels)
    loss_recon = recon_crit(fake_audio, audio_feats)
    loss_kl = kl_divergence(fake_audio)
    loss_seqdis = sequence_discrepancy(audio_feats, fake_audio)
    loss_gen = loss_recon + 0.01 * loss_adv + 0.01 * loss_kl + 0.01 * loss_seqdis

    optim_trans.zero_grad()
    loss_gen.backward()
    optim_trans.step()

    return loss_gen.item(), loss_disc.item()


def cycle_consistency_loss(original_input, translated_back):
    return F.mse_loss(original_input, translated_back)


def train_full_translation_cycle(mod1_feats, mod2_feats, translator_fwd, translator_bwd,
                                  disc, opt_trans_fwd, opt_trans_bwd, opt_disc,
                                  recon_crit, adv_crit):

    fake_mod2, latent1 = translator_fwd(mod1_feats)
    cycle_mod1, latent2 = translator_bwd(fake_mod2)

    disc.train()
    real_labels = torch.ones(mod2_feats.size(0), 1)
    fake_labels = torch.zeros(mod2_feats.size(0), 1)
    pred_real = disc(mod2_feats.detach())
    pred_fake = disc(fake_mod2.detach())
    loss_disc = (adv_crit(pred_real, real_labels) + adv_crit(pred_fake, fake_labels)) / 2
    opt_disc.zero_grad(); loss_disc.backward(); opt_disc.step()

    disc.eval()
    pred_fake = disc(fake_mod2)
    loss_adv = adv_crit(pred_fake, real_labels)
    loss_recon = recon_crit(fake_mod2, mod2_feats)
    loss_cycle = cycle_consistency_loss(mod1_feats, cycle_mod1)
    total_loss = loss_recon + 0.01 * loss_adv + 0.01 * loss_cycle

    opt_trans_fwd.zero_grad()
    opt_trans_bwd.zero_grad()
    total_loss.backward()
    opt_trans_fwd.step()
    opt_trans_bwd.step()

    return total_loss.item(), loss_disc.item(), loss_cycle.item()


# In[65]:


# 5. TRAINING LOOP & INFERENCE FUNCTIONS
# ===========================================
def generate_audio_from_text(text_feats, translator):
    translator.eval()
    with torch.no_grad():
        recon_audio, _ = translator(text_feats)
    return recon_audio


def classify_with_generated_audio(model, text_feats, visual_feats, translator):
    model.eval()
    with torch.no_grad():
        fake_audio = generate_audio_from_text(text_feats, translator)
        fake_audio_seq = fake_audio.unsqueeze(1).repeat(1, text_feats.shape[1], 1)
        preds = model(fake_audio_seq, visual_feats, text_feats)
    return preds


def train_emotion_classifier(model, dataloader, translator, criterion, optimizer, gan_args):
    model.train()
    total_classification_loss = 0
    total_gan_loss = 0

    for audio, visual, text, label in dataloader:
        optimizer.zero_grad()

        t_avg = text.mean(dim=1)
        a_avg = audio.mean(dim=1)
        v_avg = visual.mean(dim=1)

        # Generate fake audio from text
        fake_audio = generate_audio_from_text(t_avg, translator)
        fake_audio_seq = fake_audio.unsqueeze(1).repeat(1, text.shape[1], 1)
        visual_seq = v_avg.unsqueeze(1).repeat(1, text.shape[1], 1)

        # Forward pass
        outputs = model(fake_audio_seq, visual_seq, text)
        loss = criterion(outputs, label)
        loss.backward()
        optimizer.step()
        total_classification_loss += loss.item()

        # Train GAN (text → audio)
        gan_loss, _ = train_cross_modal_gan(
            t_avg, a_avg,
            gan_args['translator'], gan_args['discriminator'],
            gan_args['translator_optim'], gan_args['discriminator_optim'],
            gan_args['recon_loss'], gan_args['adv_loss']
        )
        total_gan_loss += gan_loss

    return total_classification_loss, total_gan_loss


# In[67]:


# 6. EVALUATION & CONFUSION MATRIX
# ===========================================
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

def evaluate_model(model, dataloader, translator, show_confusion=True):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for audio, visual, text, label in dataloader:
            t_avg = text.mean(dim=1)
            v_avg = visual.mean(dim=1)

            fake_audio = generate_audio_from_text(t_avg, translator)
            fake_audio_seq = fake_audio.unsqueeze(1).repeat(1, text.shape[1], 1)
            visual_seq = v_avg.unsqueeze(1).repeat(1, text.shape[1], 1)

            outputs = model(fake_audio_seq, visual_seq, text)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(label.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    print(f"Evaluation Accuracy: {acc:.4f}")
    print(f"F1 Score: {f1:.4f}")

    if show_confusion:
        cm = confusion_matrix(all_labels, all_preds, normalize='true')
        fig, ax = plt.subplots(figsize=(6, 5))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap='Blues', ax=ax, values_format='.2f')
        ax.set_title('Confusion Matrix')
        ax.set_xlabel('Predicted label')
        ax.set_ylabel('True label')
        plt.grid(False)
        plt.tight_layout()
        plt.show()

    return acc, f1


# In[69]:


# 7. ADVANCED EXPERIMENTS & ABLATION MODULES
# ===========================================
def evaluate_real_vs_generated(model, dataloader, translator):
    print("Evaluating: Real vs Generated Modality Comparison")
    model.eval()
    all_real_preds, all_gen_preds, all_labels = [], [], []

    with torch.no_grad():
        for audio, visual, text, label in dataloader:
            t_avg = text.mean(dim=1)
            v_avg = visual.mean(dim=1)

            # Real audio
            out_real = model(audio, visual, text)
            pred_real = torch.argmax(out_real, dim=1)
            all_real_preds.extend(pred_real.cpu().numpy())

            # Generated audio
            fake_audio = generate_audio_from_text(t_avg, translator)
            fake_audio_seq = fake_audio.unsqueeze(1).repeat(1, text.shape[1], 1)
            visual_seq = v_avg.unsqueeze(1).repeat(1, text.shape[1], 1)
            out_fake = model(fake_audio_seq, visual_seq, text)
            pred_fake = torch.argmax(out_fake, dim=1)
            all_gen_preds.extend(pred_fake.cpu().numpy())

            all_labels.extend(label.cpu().numpy())

    from sklearn.metrics import accuracy_score
    real_acc = accuracy_score(all_labels, all_real_preds)
    gen_acc = accuracy_score(all_labels, all_gen_preds)

    print(f"Real Input Accuracy: {real_acc:.4f}")
    print(f"Generated Input Accuracy: {gen_acc:.4f}")
    return real_acc, gen_acc


def evaluate_with_dropout(model, dataloader, dropout_modality='audio'):
    print(f"Evaluating with {dropout_modality.upper()} modality dropped")
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for audio, visual, text, label in dataloader:
            if dropout_modality == 'audio':
                audio = torch.zeros_like(audio)
            elif dropout_modality == 'visual':
                visual = torch.zeros_like(visual)
            elif dropout_modality == 'text':
                text = torch.zeros_like(text)

            outputs = model(audio, visual, text)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(label.cpu().numpy())

    from sklearn.metrics import accuracy_score
    acc = accuracy_score(all_labels, all_preds)
    print(f"Accuracy with {dropout_modality.upper()} dropped: {acc:.4f}")
    return acc


def initialize_and_train_all_with_eval():
    model = initialize_and_train_all()
    dataset = MultimodalDataset(data_df, DummyAudioExtractor(), DummyVisualExtractor(), TextExtractor())
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False)

    print("\n--- Running Advanced Evaluations ---")
    evaluate_real_vs_generated(model, dataloader, CrossModalTranslator(input_dim=768, output_dim=300))
    evaluate_with_dropout(model, dataloader, 'audio')
    evaluate_with_dropout(model, dataloader, 'visual')
    evaluate_with_dropout(model, dataloader, 'text')


# In[71]:


# 8. FULL EXECUTION & RESULTS VISUALIZATION
# ===========================================
def full_run():
    print("Initializing model and components...")

    # Extractors
    audio_ext = DummyAudioExtractor()
    visual_ext = DummyVisualExtractor()
    text_ext = TextExtractor()

    # Dataset
    dataset = MultimodalDataset(data_df, audio_ext, visual_ext, text_ext)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    # Model & Loss
    model = EmotionClassifier()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # GAN Components
    text_to_audio = CrossModalTranslator(input_dim=768, output_dim=300)
    translator_optim = torch.optim.Adam(text_to_audio.parameters(), lr=1e-4)
    discriminator = Discriminator(input_dim=300)
    discriminator_optim = torch.optim.Adam(discriminator.parameters(), lr=1e-4)
    adv_loss = nn.BCELoss()
    recon_loss = nn.MSELoss()

    classification_losses = []
    reconstruction_losses = []

    print("\nTraining on CMU-MOSEI and IEMOCAP merged dataset...")
    for epoch in range(3):
        loss_cls, loss_gan = train_emotion_classifier(
            model, dataloader, text_to_audio, criterion, optimizer,
            gan_args={
                'translator': text_to_audio,
                'discriminator': discriminator,
                'translator_optim': translator_optim,
                'discriminator_optim': discriminator_optim,
                'adv_loss': adv_loss,
                'recon_loss': recon_loss
            }
        )
        classification_losses.append(loss_cls)
        reconstruction_losses.append(loss_gan)
        print(f"Epoch {epoch+1}: Classification Loss = {loss_cls:.4f}, GAN Loss = {loss_gan:.4f}")

    # Evaluation
    print("\nFinal Evaluation:")
    split_set = MultimodalDataset(data_df, audio_ext, visual_ext, text_ext)
    split_loader = DataLoader(split_set, batch_size=4, shuffle=False)
    evaluate_model(model, split_loader, text_to_audio)

    # Plot
    epochs = list(range(1, len(classification_losses) + 1))
    plt.figure(figsize=(12, 6))
    plt.plot(epochs, classification_losses, marker='o', color='royalblue', linewidth=2, label='Classification Loss')
    plt.plot(epochs, reconstruction_losses, marker='s', color='orangered', linewidth=2, label='Reconstruction Loss')
    plt.title('Training Performance Over Epochs', fontsize=16, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.xticks(epochs)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.show()


# To execute everything
if __name__ == "__main__":
    full_run()


# In[73]:





# In[1]:


get_ipython().system('jupyter nbconvert --to script your_notebook.ipynb')


# In[ ]:




