import math


def calc_mi(z, mu, logv):
	"""Approximate the mutual information between x and z
	 I(x, z) = E_xE_{q(z|x)}log(q(z|x)) - E_xE_{q(z|x)}log(q(z))
	 Returns: Float
	 """

	nz=z.shape[-1]
	bs=z.shape[1]

	# E_{q(z|x)}log(q(z|x)) = -0.5*nz*log(2*\pi) - 0.5*(1+logvar).sum(-1)
	neg_entropy = (
			-0.5 * nz * math.log(2 * math.pi) - 0.5 * (1 + logvar).sum(-1)
	).mean()

	# [z_batch, 1, nz]
	# z_samples = model.t5.reparameterize(mu, logvar)
	# z_samples = z_samples.unsqueeze(1)

	# [1, x_batch, nz]
	mu, logvar = mu.unsqueeze(0), logvar.unsqueeze(0)
	var = logvar.exp()

	# (z_batch, x_batch, nz)
	dev = z_samples - mu

	# (z_batch, x_batch)
	log_density = -0.5 * ((dev ** 2) / var).sum(dim=-1) - 0.5 * (
			nz * math.log(2 * math.pi) + logvar.sum(-1)
	)

	# log q(z): aggregate posterior
	# [z_batch]
	log_qz = log_sum_exp(log_density, dim=1) - math.log(x_batch)

	return (neg_entropy - log_qz.mean(-1)).item()
