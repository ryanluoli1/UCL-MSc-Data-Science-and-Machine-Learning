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
        self.classifier = nn.Linear(in_features=int(params['txt_embed_size']),
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
        # temporal average as global representation
        x_txt = torch.mean(x_txt, dim=1)
        # classification
        y = self.classifier(x_txt)
        return y, x_txt


class SSE_Text_CLS(nn.Module):
    """
    Unimodal (text) self-attention model with self-supervised embeddings.
    """

    def __init__(self, params):
        super(SSE_Text_CLS, self).__init__()
        # multi-head self-attention
        self.mha_t = nn.MultiheadAttention(embed_dim=params['txt_embed_size'],
                                           num_heads=params['txt_H'],
                                           dropout=params['txt_dropout'],
                                           batch_first=True)
        # fully-connected layers
        self.classifier = nn.Linear(in_features=int(params['txt_embed_size']),
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
        # CLS token as global representation
        x_txt = x_txt[:, 0, :]
        # classification
        y = self.classifier(x_txt)
        return y, x_txt


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
        self.classifier = nn.Linear(in_features=int(params['embed_size']),
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
        # temporal average as global representation
        x_aud = torch.mean(x_aud, dim=1)
        # classification
        y = self.classifier(x_aud)
        return y, x_aud

