import math
import torch

def log_sum_exp(value, dim=None, keepdim=False):
    """Numerically stable implementation of the operation
    value.exp().sum(dim, keepdim).log()
    """
    if dim is not None:
        m, _ = torch.max(value, dim=dim, keepdim=True)
        value0 = value - m
        if keepdim is False:
            m = m.squeeze(dim)
        return m + torch.log(torch.sum(torch.exp(value0), dim=dim, keepdim=keepdim))
    else:
        m = torch.max(value)
        sum_exp = torch.sum(torch.exp(value - m))
        return m + torch.log(sum_exp)

def calc_mi(z, mu, logv):
	"""Approximate the mutual information between x and z
	 I(x, z) = E_xE_{q(z|x)}log(q(z|x)) - E_xE_{q(z|x)}log(q(z))
	 Returns: Float
	 """

	nz=z.shape[2]
	bs=z.shape[1]

	# E_{q(z|x)}log(q(z|x)) = -0.5*nz*log(2*\pi) - 0.5*(1+logvar).sum(-1)
	neg_entropy = (
			-0.5 * nz * math.log(2 * math.pi) - 0.5 * (logv).sum(-1)
	).mean()

	# [z_batch, 1, nz]
	# z_samples = model.t5.reparameterize(mu, logvar)
	# z_samples = z_samples.unsqueeze(1)

	# [1, x_batch, nz]
	# print(mu.shape)
	# mu, logvar = mu.unsqueeze(0), logvar.unsqueeze(0)
	var = logv.exp()

	# (z_batch, x_batch, nz)
	dev = z - mu

	# (z_batch, x_batch)
	log_density = -0.5 * ((dev ** 2) / var).sum(dim=-1) - 0.5 * (
			nz * math.log(2 * math.pi) + logv.sum(-1)
	)

	# log q(z): aggregate posterior
	# [z_batch]
	log_qz = log_sum_exp(log_density, dim=1) - math.log(bs)

	return (neg_entropy - log_qz.mean(-1)).item()