import torch
import torch.nn as nn
import itertools

torch.set_printoptions(profile="full")


class EDEE(nn.Module):
    def __init__(self,args,word_type_tag_num):
        super(EDEE, self).__init__()
        self.args = args

        num_embeddings, embed_dim = args.token_embedding.shape
        self.embed = nn.Embedding(num_embeddings, embed_dim)
        self.embed.weight = nn.Parameter(args.token_embedding,requires_grad=False)

        self.word_type_embed = nn.Embedding(word_type_tag_num, args.word_type_embedding_dim)

        self.dropout = nn.Dropout(args.dropout)


        in_dim = args.word_embedding_dim+args.word_type_embedding_dim
        self.token_bilstm = nn.LSTM(input_size=in_dim, hidden_size=args.hidden_size,
                              bidirectional=True, batch_first=True, num_layers=args.num_layers)

        last_hidden_size = 4*args.hidden_size
        layers = [nn.Linear(last_hidden_size, args.final_hidden_size), nn.LeakyReLU()]
        for _ in range(args.num_mlps - 1):
            layers += [nn.Linear(args.final_hidden_size,
                                 args.final_hidden_size), nn.LeakyReLU()]
        self.fcs = nn.Sequential(*layers)
        self.fc_final = nn.Linear(args.final_hidden_size, args.role_role_num)

    def forward(self,word_ids,wType_ids):
        token_feature = self.embed(word_ids)
        token_feature = self.dropout(token_feature)
        token_type_feature = self.word_type_embed(wType_ids)
        token_type_feature = self.dropout(token_type_feature)

        all_token_feature = torch.cat([token_feature,token_type_feature],dim=1)

        token_out_bilstm, _ = self.token_bilstm(all_token_feature.unsqueeze(0))
        token_out_bilstm = self.dropout(token_out_bilstm).squeeze(0)

        ent_ent_list = list(itertools.product(token_out_bilstm,repeat=2))
        ent_ent_emb = []
        for ent_ent in ent_ent_list:
            ent_ent_emb.append(torch.cat([ent_ent[0],ent_ent[1]],dim=0))

        ent_ent_feature = torch.stack(ent_ent_emb,dim=0)

        out = self.fcs(ent_ent_feature)
        logits = self.fc_final(out)

        return logits



