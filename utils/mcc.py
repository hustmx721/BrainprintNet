import torch
args = None


if args.mcc==True:
    def Entropy(input_):
        bs = input_.size(0)
        epsilon = 1e-5
        entropy = -input_ * torch.log(input_ + epsilon)
        entropy = torch.sum(entropy, dim=1)
        return entropy

    unsup_x_eeg =unsup_batch[0].cuda()
    unsup_output_s,_=model(unsup_x_eeg) 
    output_s_temp = unsup_output_s/2.5  # T
    s_softmax_out_temp = nn.Softmax(dim=1)(output_s_temp)
    s_entropy_weight = Entropy(s_softmax_out_temp).detach()
    s_entropy_weight = 1 + torch.exp(-s_entropy_weight)
    s_entropy_weight = 32* s_entropy_weight / torch.sum(s_entropy_weight) # bs
    cov_matrix_t = s_softmax_out_temp.mul(s_entropy_weight.view(-1,1)).transpose(1,0).mm(s_softmax_out_temp)
    cov_matrix_t = cov_matrix_t / torch.sum(cov_matrix_t, dim=1)
    loss_mcc = (torch.sum(cov_matrix_t) - torch.trace(cov_matrix_t)) / 5 # nclass
else:
    loss_mcc=0