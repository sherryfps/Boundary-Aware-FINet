import math


class CosineDecay:
	def __init__(self,
	             optimizer,
	             max_lr,
	             min_lr,
	             max_epoch,
	             test_mode=False):
		self.optimizer = optimizer
		self.max_lr = max_lr
		self.min_lr = min_lr
		self.max_epoch = max_epoch
		self.test_mode = test_mode

		self.current_lr = max_lr
		self.cnt = 0
		if self.max_epoch < 1:
			raise ValueError('max_epoch must be positive')
		# A one-epoch smoke run previously left these attributes undefined.
		self.scale = 0
		self.shift = max_lr
		self.alpha = 0
		if self.max_epoch > 1:
			self.scale = (max_lr - min_lr) / 2
			self.shift = (max_lr + min_lr) / 2
			self.alpha = math.pi / (max_epoch - 1)

	def step(self):
		self.cnt += 1
		self.current_lr = self.scale * math.cos(self.alpha * self.cnt) + self.shift

		if not self.test_mode:
			for param_group in self.optimizer.param_groups:
				param_group['lr'] = self.current_lr

	def get_lr(self):
		return self.current_lr

	def state_dict(self):
		return {'max_lr': self.max_lr, 'min_lr': self.min_lr,
		        'max_epoch': self.max_epoch, 'cnt': self.cnt,
		        'current_lr': self.current_lr}

	def load_state_dict(self, state):
		for key in ('max_lr', 'min_lr', 'max_epoch'):
			if state[key] != getattr(self, key):
				raise ValueError(f'scheduler configuration mismatch: {key}')
		if state['cnt'] < 0 or not math.isfinite(state['current_lr']):
			raise ValueError('invalid scheduler state')
		self.cnt = state['cnt']
		self.current_lr = state['current_lr']
		if not self.test_mode:
			for group in self.optimizer.param_groups:
				group['lr'] = self.current_lr


if __name__ == '__main__':
	import matplotlib.pyplot as plt

	max_epoch = 100
	s = CosineDecay(None, max_lr=2.5e-4, min_lr=1e-7, max_epoch=max_epoch, test_mode=True)
	lr_list = []
	for i in range(max_epoch):
		lr_list.append(s.get_lr())
		s.step()

	plt.plot(list(range(max_epoch)), lr_list)
	plt.show()
