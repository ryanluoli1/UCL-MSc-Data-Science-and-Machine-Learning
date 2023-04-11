import torch
import torch.nn as nn

class SSE_Text(nn.Module):
    """
    Unimodal (text) self-attention model with self-supervised embeddings.
    """

    def __init__(self, params):
        super(SSE_Text, self).__init__()
        # multi-head self-attention
        self.mha_t = nn.MultiheadAttention(embed_dim=params['txt_embed_size'],
                                           num_heads=params['txt_H'],
                                           dropout=params['txt_dropout'],
                                           batch_first=True)
        # fully-connected layers
        self.concat_linear = nn.Linear(in_features=int(2 * params['txt_embed_size']),
                                       out_features=int(0.5 * params['txt_embed_size']))
        self.classifier = nn.Linear(in_features=int(0.5 * params['txt_embed_size']),
                                    out_features=params['output_dim'])

    def forward(self, x_txt):
        """
        Input:
            x_txt   | text embeddings from RoBERTa, embedding size = 1024
        Output:
            y       | unnormalized logits, shape = (batch_size, n_class)
            x_t     | text features, shape = (batch_size, 384)
        """
        # multi-head self-attention
        x_txt, _ = self.mha_t(x_txt, x_txt, x_txt)
        # statistical pooling
        x_std, x_mean = torch.std_mean(x_txt, dim=1)
        x_mean_std = torch.cat((x_mean, x_std), dim=1)
        features = self.concat_linear(x_mean_std)
        # classification
        y = self.classifier(features)
        return y, features


class SSE_Audio(nn.Module):
    """
    Unimodal (audio) self-attention model with self-supervised embeddings.
    """

    def __init__(self, params):
        super(SSE_Audio, self).__init__()
        # multi-head self-attention
        self.mha_a = nn.MultiheadAttention(embed_dim=params['embed_size'],
                                           num_heads=params['aud_H'],
                                           dropout=params['aud_dropout'],
                                           batch_first=True)
        # fully-connected layers
        self.concat_linear = nn.Linear(in_features=int(2 * params['embed_size']),
                                       out_features=int(0.5 * params['embed_size']))
        self.classifier = nn.Linear(in_features=int(0.5 * params['embed_size']),
                                    out_features=params['output_dim'])

    def forward(self, x_aud):
        """
        Input:
            x_aud   | audio embeddings from Wav2Vec 2.0, embeddings size = 768
        Output:
            y       | unnormalized logits, shape = (batch_size, n_class)
            x_ta    | audio features, shape = (batch_size, 384)
        """
        # multi-head self-attention
        x_aud, _ = self.mha_a(x_aud, x_aud, x_aud)
        # statistical pooling
        x_std, x_mean = torch.std_mean(x_aud, dim=1)
        x_mean_std = torch.cat((x_mean, x_std), dim=1)
        features = self.concat_linear(x_mean_std)
        # classification
        y = self.classifier(features)
        return y, features


class SSE_Self(nn.Module):
    """
    Bimodal (audio & text) self-attention model with self-supervised embeddings.
    """

    def __init__(self, params):
        super(SSE_Self, self).__init__()
        # multi-head self-attention
        self.mha_t = nn.MultiheadAttention(embed_dim=params['embed_size'],
                                           num_heads=params['txt_H'],
                                           dropout=params['txt_dropout'],
                                           batch_first=True)
        self.mha_a = nn.MultiheadAttention(embed_dim=params['embed_size'],
                                           num_heads=params['aud_H'],
                                           dropout=params['aud_dropout'],
                                           batch_first=True)
        # fully-connected layers
        self.dim_reduction = nn.Linear(in_features=params['txt_embed_size'],
                                       out_features=params['embed_size'])
        self.concat_linear = nn.Linear(in_features=int(2 * params['embed_size']),
                                       out_features=int(0.5 * params['embed_size']))
        self.classifier = nn.Linear(in_features=int(0.5 * params['embed_size']),
                                    out_features=params['output_dim'])

    def forward(self, x_txt, x_aud):
        """
        Input:
            x_txt   | text embeddings from RoBERTa, embedding size = 1024
            x_aud   | audio embeddings from Wav2Vec 2.0, embeddings size = 768
        Output:
            y       | unnormalized logits, shape = (batch_size, n_class)
            x_ta    | fused text-audio features, shape = (batch_size, 384)
        """
        # dimension reduction
        x_txt = self.dim_reduction(x_txt)
        # multi-head self-attention
        x_txt, _ = self.mha_t(x_txt, x_txt, x_txt)
        x_aud, _ = self.mha_a(x_aud, x_aud, x_aud)
        # temporal average as global representation
        x_txt_avg = torch.mean(x_txt, dim=1)
        x_aud_avg = torch.mean(x_aud, dim=1)
        # concatenating text and audio features
        x_ta = torch.stack((x_txt_avg, x_aud_avg), dim=1)
        # statistical pooling
        x_ta_std, x_ta_mean = torch.std_mean(x_ta, dim=1)
        x_ta = torch.cat((x_ta_mean, x_ta_std), dim=1)
        x_ta = self.concat_linear(x_ta)
        # classification
        y = self.classifier(x_ta)
        return y, x_ta


class SSE_Cross(nn.Module):
    """
    Bimodal (audio & text) cross-attention model with self-supervised embeddings.
    """
    def __init__(self, params):
        super(SSE_Cross, self).__init__()
        # multi-head cross-attention
        self.mha_ta = nn.MultiheadAttention(embed_dim=params['embed_size'],
                                            num_heads=params['ta_H'],
                                            dropout=params['ta_dropout'],
                                            batch_first=True)
        self.mha_at = nn.MultiheadAttention(embed_dim=params['embed_size'],
                                            num_heads=params['at_H'],
                                            dropout=params['at_dropout'],
                                            batch_first=True)
        # fully-connected layers
        self.dim_reduction = nn.Linear(in_features=params['txt_embed_size'],
                                       out_features=params['embed_size'])
        self.concat_linear = nn.Linear(in_features=int(2 * params['embed_size']),
                                       out_features=int(0.5 * params['embed_size']))
        self.classifier = nn.Linear(in_features=int(0.5 * params['embed_size']),
                                    out_features=params['output_dim'])

    def forward(self, x_txt, x_aud):
        """
        Input:
            x_txt   | text embeddings from RoBERTa, embedding size = 1024
            x_aud   | audio embeddings from Wav2Vec 2.0, embeddings size = 768
        Output:
            y       | unnormalized logits, shape = (batch_size, n_class)
            x_ta    | fused text-audio features, shape = (batch_size, 384)
        """
        # dimension reduction
        x_txt = self.dim_reduction(x_txt)
        # multi-head cross-attention
        x_t2a, _ = self.mha_ta(x_txt, x_aud, x_aud)
        x_a2t, _ = self.mha_at(x_aud, x_txt, x_txt)
        # temporal average as global representation
        x_t2a_avg = torch.mean(x_t2a, dim=1)
        x_a2t_avg = torch.mean(x_a2t, dim=1)
        # concatenating text and audio features
        x_ta = torch.stack((x_t2a_avg, x_a2t_avg), dim=1)
        # statistical pooling
        x_ta_std, x_ta_mean = torch.std_mean(x_ta, dim=1)
        x_ta = torch.cat((x_ta_mean, x_ta_std), dim=1)
        x_ta = self.concat_linear(x_ta)
        # classification
        y = self.classifier(x_ta)
        return y, x_ta