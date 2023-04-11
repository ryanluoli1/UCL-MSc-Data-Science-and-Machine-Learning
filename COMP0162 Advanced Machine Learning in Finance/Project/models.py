import torch
import torch.nn as nn



class Image_CNN(nn.Module):
    """
    CNN encoder for feature extraction from raw image data.
    """
    def __init__(self):
        super(Image_CNN, self).__init__()
        # convolutional layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3,3), padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(3,3), padding=1)
        # functional layers
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(p=0.01)
        self.maxpool = nn.MaxPool2d(kernel_size=(2,2))
        # classifier
        self.classifier = nn.Linear(16*25*25, 1)

    def forward(self, x_img_raw):
        """
        Input:
            x_img_raw   | raw image data, shape = (batch_size, width, length, n_channels)
        Output:
            y           | unnormalized logits, shape = (batch_size, 1)
        """
        x = self.conv1(x_img_raw)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.dropout(x)
        x = self.flatten(x)
        x = self.dropout(x)
        y = self.classifier(x)
        return y



class Image_SSE(nn.Module):
    """
    Unimodal (image) classifier based on self-supervised embeddings.
    """
    def __init__(self, params):
        super(Image_SSE, self).__init__()
        # functional layers
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=params['img_dropout'])
        # fully connected layers
        self.reduce_dim = nn.Linear(in_features=params['img_embed_size'],
                                    out_features=params['hidden_size'])
        self.classifier = nn.Linear(in_features=params['hidden_size'],
                                    out_features=1)

    def forward(self, x_img):
        """
        Input:
            x_img   | image self-supervised embeddings, shape = (batch_size, img_embedding_size)
        Output:
            y       | unnormalized logits, shape = (batch_size, 1)
        """
        # dimension reduction
        x_img = self.relu(self.reduce_dim(x_img))
        # dropout
        x_img = self.dropout(x_img)
        # classification
        y = self.classifier(x_img)
        return y



class Text_SSE(nn.Module):
    """
    Unimodal (text) classifier based on self-supervised embeddings.
    """
    def __init__(self, params):
        super(Text_SSE, self).__init__()
        # functional layers
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=params['txt_dropout'])
        # fully connected layers
        self.reduce_dim = nn.Linear(in_features=params['txt_embed_size'],
                                    out_features=params['hidden_size'])
        self.classifier = nn.Linear(in_features=params['hidden_size'],
                                    out_features=1)

    def forward(self, x_txt):
        """
        Input:
            x_txt   | text self-supervised embeddings, shape = (batch_size, txt_embedding_size)
        Output:
            y       | unnormalized logits, shape = (batch_size, 1)
        """
        # dimension reduction
        x_txt = self.relu(self.reduce_dim(x_txt))
        # dropout
        x_txt = self.dropout(x_txt)
        # temporal average as global representation
        x_txt = torch.mean(x_txt, dim=1)
        # classification
        y = self.classifier(x_txt)
        return y



class Text_SA(nn.Module):
    """
    Unimodal (text) classifier based on self-supervised embeddings and self-attention.
    """
    def __init__(self, params):
        super(Text_SA, self).__init__()
        # functional layers
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=params['txt_dropout'])
        # multi-heads self-attention
        self.mha = nn.MultiheadAttention(embed_dim=params['hidden_size'],
                                         num_heads=params['txt_H'],
                                         dropout=params['txt_dropout'],
                                         batch_first=True)
        # fully connected layers
        self.reduce_dim = nn.Linear(in_features=params['txt_embed_size'],
                                    out_features=params['hidden_size'])
        self.classifier = nn.Linear(in_features=params['hidden_size'],
                                    out_features=1)

    def forward(self, x_txt):
        """
        Input:
            x_txt   | text self-supervised embeddings, shape = (batch_size, txt_embedding_size)
        Output:
            y       | unnormalized logits, shape = (batch_size, 1)
        """
        # dimension reduction
        x_txt = self.relu(self.reduce_dim(x_txt))
        # dropout
        x_txt = self.dropout(x_txt)
        # self-attention
        x_txt, _ = self.mha(x_txt, x_txt, x_txt)
        # temporal average as global representation
        x_txt = torch.mean(x_txt, dim=1)
        # classification
        y = self.classifier(x_txt)
        return y



class Text_Image_CA(nn.Module):
    """
    Bimodal (Text+Image) classifier based on self-supervised embeddings and cross-attention.
    """
    def __init__(self, params):
        super(Text_Image_CA, self).__init__()
        # functional layers
        self.relu = nn.ReLU()
        self.dropout_txt = nn.Dropout(p=params['txt_dropout'])
        self.dropout_img = nn.Dropout(p=params['img_dropout'])
        # multi-heads cross-attention
        self.mha_vt = nn.MultiheadAttention(embed_dim=params['hidden_size'],
                                            num_heads=params['vt_H'],
                                            dropout=params['vt_dropout'],
                                            batch_first=True)
        # fully connected layers
        self.reduce_dim_txt = nn.Linear(in_features=params['txt_embed_size'],
                                        out_features=params['hidden_size'])
        self.reduce_dim_img = nn.Linear(in_features=params['img_embed_size'],
                                        out_features=params['hidden_size'])
        self.classifier = nn.Linear(in_features=params['hidden_size'],
                                    out_features=1)

    def forward(self, x_txt, x_img):
        """
        Input:
            x_txt   | text  self-supervised embeddings, shape = (batch_size, txt_embedding_size)
            x_img   | image self-supervised embeddings, shape = (batch_size, img_embedding_size)
        Output:
            y       | unnormalized logits, shape = (batch_size, 1)
            x_vt    | fused image-text features, shape = (batch_size, hidden_size)
        """
        # dimension reduction
        x_txt = self.relu(self.reduce_dim_txt(x_txt))
        x_img = self.relu(self.reduce_dim_img(x_img))
        # dropout
        x_txt = self.dropout_txt(x_txt)
        x_img = self.dropout_img(x_img)
        # inter-modality cross-attention
        x_vt, _ = self.mha_vt(x_img.unsqueeze(dim=1), x_txt, x_txt)
        # classification
        y = self.classifier(x_vt.squeeze())
        return y



class Text_Image_SA(nn.Module):
    """
    Bimodal (Text+Image) classifier based on self-supervised embeddings and self-attention.
    """
    def __init__(self, params):
        super(Text_Image_SA, self).__init__()
        # functional layers
        self.relu = nn.ReLU()
        self.dropout_txt = nn.Dropout(p=params['txt_dropout'])
        self.dropout_img = nn.Dropout(p=params['img_dropout'])
        self.dropout_fused = nn.Dropout(p=params['fused_dropout'])
        # multi-heads cross-attention
        self.mha_txt = nn.MultiheadAttention(embed_dim=params['hidden_size'],
                                             num_heads=params['txt_H'],
                                             dropout=params['txt_dropout'],
                                             batch_first=True)
        # fully connected layers
        self.reduce_dim_txt = nn.Linear(in_features=params['txt_embed_size'],
                                        out_features=params['hidden_size'])
        self.reduce_dim_img = nn.Linear(in_features=params['img_embed_size'],
                                        out_features=params['hidden_size'])
        self.concat_linear = nn.Linear(in_features=int(2*params['hidden_size']),
                                       out_features=int(0.5*params['hidden_size']))
        self.classifier = nn.Linear(in_features=int(0.5*params['hidden_size']),
                                    out_features=1)

    def forward(self, x_txt, x_img):
        """
        Input:
            x_txt   | text  self-supervised embeddings, shape = (batch_size, txt_embedding_size)
            x_img   | image self-supervised embeddings, shape = (batch_size, img_embedding_size)
        Output:
            y       | unnormalized logits, shape = (batch_size, 1)
            x_vt    | fused image-text features, shape = (batch_size, hidden_size)
        """
        # dimension reduction
        x_txt = self.relu(self.reduce_dim_txt(x_txt))
        x_img = self.relu(self.reduce_dim_img(x_img))
        # dropout
        x_txt = self.dropout_txt(x_txt)
        x_img = self.dropout_img(x_img)
        # self-attention
        x_txt, _ = self.mha_txt(x_txt, x_txt, x_txt)
        # temporal average as global representation
        x_txt = torch.mean(x_txt, dim=1)
        # fusion with statistical pooling
        x_fused = torch.stack((x_txt, x_img), dim=1)
        x_std, x_mean = torch.std_mean(x_fused, dim=1)
        x_fused = torch.cat((x_std, x_mean), dim=1)
        x_fused = self.dropout_fused(self.concat_linear(x_fused))
        # classification
        y = self.classifier(x_fused)
        return y



class Text_Image_Transformer(nn.Module):
    """
    Bimodal (Text+Image) classifier based on self-supervised embeddings and cross-attention.
    """

    def __init__(self, params):
        super(Text_Image_Transformer, self).__init__()
        # multi-heads cross-attention
        self.mha_vt = nn.MultiheadAttention(embed_dim=params['hidden_size'],
                                            num_heads=params['vt_H'],
                                            dropout=params['vt_dropout'],
                                            batch_first=True)

        # point-to-point feed-forward network
        self.feedforward = nn.Sequential(nn.Linear(params['hidden_size'], 4 * params['hidden_size']),
                                         nn.ReLU(),
                                         nn.Linear(4 * params['hidden_size'], params['hidden_size']))

        # functional layers
        self.relu = nn.ReLU()
        self.norm = nn.LayerNorm(params['hidden_size'])
        self.dropout_txt = nn.Dropout(p=params['txt_dropout'])
        self.dropout_img = nn.Dropout(p=params['img_dropout'])
        self.dropout = nn.Dropout(params['transformer_dropout'])

        # fully connected layers
        self.reduce_dim_txt = nn.Linear(in_features=params['txt_embed_size'],
                                        out_features=params['hidden_size'])
        self.reduce_dim_img = nn.Linear(in_features=params['img_embed_size'],
                                        out_features=params['hidden_size'])
        self.classifier = nn.Linear(in_features=params['hidden_size'],
                                    out_features=1)

    def forward(self, x_txt, x_img):
        """
        Input:
            x_txt   | text  self-supervised embeddings, shape = (batch_size, txt_embedding_size)
            x_img   | image self-supervised embeddings, shape = (batch_size, img_embedding_size)
        Output:
            y       | unnormalized logits, shape = (batch_size, 1)
            x_vt    | fused image-text features, shape = (batch_size, hidden_size)
        """
        # dimension reduction
        x_txt = self.relu(self.reduce_dim_txt(x_txt))
        x_img = self.relu(self.reduce_dim_img(x_img))
        # dropout
        x_txt = self.dropout_txt(x_txt)
        x_img = self.dropout_img(x_img)
        # inter-modality cross-attention
        x_vt, _ = self.mha_vt(x_img.unsqueeze(dim=1), x_txt, x_txt)
        # residual connection and layer norm
        x_vt = self.norm(self.dropout(x_vt) + x_img.unsqueeze(dim=1))
        # feed-forward network
        x_vt_ffn = self.feedforward(x_vt)
        # residual connection and layer norm
        x_vt = self.norm(self.dropout(x_vt_ffn) + x_vt)
        # classification
        y = self.classifier(x_vt.squeeze())
        return y
