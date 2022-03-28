import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

MAX_TRACK_LEN = 256

class StateClassifier(nn.Module):

	def __init__(self, n_classes, pretrained=True, ftune=False, sigmoid=True, resnet50=False):
		super(StateClassifier, self).__init__()
		if resnet50:
			print("Using Resnet-50!")
			self.trunk = nn.Sequential(*list(models.resnet50(pretrained=pretrained).children())[:-1])
		else:
			self.trunk = nn.Sequential(*list(models.resnet18(pretrained=pretrained).children())[:-1])
		self.ftune = ftune
		print(f"Ftune set to {self.ftune}")
		if not self.ftune:
			for param in self.trunk.parameters():
				param.requires_grad = False

		self.head = nn.Sequential(
			nn.Linear(2048 if resnet50 else 512, n_classes),
			nn.Sigmoid() if sigmoid else nn.Identity(),
		)

	def set(self):
		if not self.ftune:
			self.trunk.eval()

	def forward(self, dic):
		images = dic['img'].clone()
		conv_features = self.trunk(images).reshape(images.shape[0], -1)
		probs = self.head(conv_features)

		dic = {'probs': probs}

		return dic

class StateClassifierwithResNet(nn.Module):

	def __init__(self, n_classes, pretrained=True, ftune=True, sigmoid=True):
		super(StateClassifierwithResNet, self).__init__()
		self.trunk = nn.Sequential(*list(models.resnet18(pretrained=pretrained).children())[:-1])

		self.resnet = nn.Sequential(*list(models.resnet34(pretrained=pretrained).children())[:-1])
		self.ftune = ftune
		print(f"Ftune set to {self.ftune}")
		if not self.ftune:
			for param in self.trunk.parameters():
				param.requires_grad = False

			for param in self.resnet.parameters():
				param.requires_grad = False

		self.pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

		self.head = nn.Sequential(
			nn.Linear(512 + 512, n_classes),
			nn.Sigmoid() if sigmoid else nn.Identity(),
			)

	def set(self):
		if not self.ftune:
			self.trunk.eval()
			self.resnet.eval()

	def forward(self, dic):
		images = dic['img'].clone()
		conv_features = self.trunk(images).reshape(images.shape[0], -1)

		res_conv_features = self.resnet(images).reshape(images.shape[0], -1)
		# conv_features = self.trunk(images)

		stacked_features = torch.cat([conv_features, res_conv_features], dim=1)

		probs = self.head(stacked_features)

		dic = {'probs': probs}

		return dic


class HeadStateClassifier(nn.Module):

	def __init__(self, feat_size, trunk, n_classes, ftune=False, sigmoid=True):
		super(HeadStateClassifier, self).__init__()
		self.trunk = trunk
		self.ftune = ftune
		print(f"Ftune set to {self.ftune}")
		if not self.ftune:
			for param in self.trunk.parameters():
				param.requires_grad = False

		self.head = nn.Sequential(
			nn.Linear(feat_size, n_classes),
			nn.Sigmoid() if sigmoid else nn.Identity(),
			)

	def set(self):
		if not self.ftune:
			self.trunk.eval()

	def forward(self, dic):
		images = dic['img'].clone()
		conv_features = self.trunk(images).reshape(images.shape[0], -1)
		# conv_features = self.trunk(images)

		probs = self.head(conv_features)

		dic = {'probs': probs}

		return dic

class HeadStateClassifierwithResNet(nn.Module):

	def __init__(self, feat_size, trunk, n_classes, pretrained=True, ftune=False, sigmoid=True):
		super(HeadStateClassifierwithResNet, self).__init__()
		self.trunk = trunk
		
		self.resnet = nn.Sequential(*list(models.resnet34(pretrained=pretrained).children())[:-1])
		self.ftune = ftune
		print(f"Ftune set to {self.ftune}")
		if not self.ftune:
			for param in self.trunk.parameters():
				param.requires_grad = False

			for param in self.resnet.parameters():
				param.requires_grad = False

		self.pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

		self.head = nn.Sequential(
			nn.Linear(feat_size + 512, n_classes),
			nn.Sigmoid() if sigmoid else nn.Identity(),
			)

	def set(self):
		if not self.ftune:
			self.trunk.eval()
			self.resnet.eval()

	def forward(self, dic):
		images = dic['img'].clone()
		conv_features = self.trunk(images).reshape(images.shape[0], -1)

		res_conv_features = self.resnet(images).reshape(images.shape[0], -1)
		# conv_features = self.trunk(images)

		stacked_features = torch.cat([conv_features, res_conv_features], dim=1)

		probs = self.head(stacked_features)

		dic = {'probs': probs}

		return dic


class FeatureExtractor(nn.Module):
	def __init__(self, feat_size, pretrained=True):
		super(FeatureExtractor, self).__init__()
		self.trunk = nn.Sequential(*list(models.resnet18(pretrained=pretrained).children())[:-1])
		self.head = nn.Sequential(
			nn.Linear(512, 512), nn.Dropout(0.0), nn.ReLU(),
			nn.Linear(512, feat_size))

	def forward(self, images):
		images = images.clone()
		conv_features = self.trunk(images).reshape(images.shape[0], -1)

		feats = self.head(conv_features)

		return feats


class FeatureExtractorMultiHead(nn.Module):
    def __init__(self, emb_size, dropout_p=0.4, num_heads=1):
        super(FeatureExtractorMultiHead, self).__init__()
        self.trunk = nn.Sequential(*list(models.resnet18(pretrained=True).children())[:-1])
        self.head = nn.ModuleList([
            nn.Sequential(
                nn.Linear(512, 512), nn.Dropout(dropout_p), nn.ReLU(),
                nn.Linear(512, emb_size), nn.ReLU()
            ) for _ in range(num_heads)
        ])

    def forward(self, images, head_idx=0):
        conv_features = self.trunk(images).reshape(images.shape[0], -1)
        return self.head[head_idx](conv_features)

class FeatureExtractorBN(nn.Module):
	def __init__(self, emb_size, pretrained=True):
		super(FeatureExtractorBN, self).__init__()
		self.trunk = nn.Sequential(*list(models.resnet18(pretrained=pretrained).children())[:-1])
		self.head = nn.Sequential(
			nn.Linear(512, 512), nn.BatchNorm1d(512), nn.Dropout(0.4), nn.ReLU(),
			nn.Linear(512, emb_size), nn.ReLU())
	def forward(self, images):
		conv_features = self.trunk(images).reshape(images.shape[0], -1)
		return self.head(conv_features)


class TCN(nn.Module):

	def __init__(self, model_config):
		super(TCN, self).__init__()
		# self.num_objects = model_config["num_objects"]
		self.feat_size = model_config["feat_size"]
		# self.num_states = model_config["num_states"]
		# self.assignment_tactic = model_config["assignment"]

		# self.sim_type = model_config["similarity"]
		# self.temperature = model_config["temperature"]

		# self.embeddings = nn.Parameter(torch.randn((
		#     self.num_states, self.num_objects, self.emb_size), requires_grad=True))

		self.extractor = FeatureExtractor(self.feat_size)

	def forward(self, inp):
		anchor, pos, neg = inp['anchor'], inp['positive'], inp['negative']

		featuresa = self.extractor(anchor)
		featuresp = self.extractor(pos)
		featuresn = self.extractor(neg)

		return {'fa': featuresa, 'fp': featuresp, 'fn': featuresn}


class ActionClassifier(nn.Module):
	def __init__(self, config):
		super(ActionClassifier, self).__init__()
		self.num_classes = len(config["data"]["filter"]["verb"])
		self.feat_size = config["model"]["feat_size"]
		# self.num_states = model_config["num_states"]
		# self.assignment_tactic = model_config["assignment"]

		# self.sim_type = model_config["similarity"]
		# self.temperature = model_config["temperature"]

		# self.embeddings = nn.Parameter(torch.randn((
		#     self.num_states, self.num_objects, self.emb_size), requires_grad=True))

		self.extractor = FeatureExtractor(self.feat_size)
		self.classifier = nn.Sequential(
									nn.Linear(self.feat_size*2, self.num_classes),
									nn.LogSoftmax(dim=1)
									)

	def forward(self, inp):
		im1, im2 = inp['img1'], inp['img2']

		features1 = self.extractor(im1)
		features2 = self.extractor(im2)
		features_stacked = torch.cat([features1, features2], dim=1)

		

		probs = self.classifier(features_stacked)

		return {'probs': probs}


class VOS(nn.Module):
	def __init__(self, model_config):
		super(VOS, self).__init__()
		self.num_objects = model_config["num_objects"]
		self.emb_size = model_config["emb_size"]
		self.num_states = model_config["num_states"]
		self.assignment_tactic = model_config["assignment"]
		self.sim_type = model_config["similarity"]
		self.temperature = model_config["temperature"]

		self.embeddings = nn.Parameter(torch.randn((
			self.num_states, self.num_objects, self.emb_size), requires_grad=True))
		self.network = FeatureExtractor(self.emb_size)

	def get_features(self, images):
		"""
		Input: [bs, MAX_TRACK_LENGTH, 3, height, width]
		Output: [MAX_TRACK_LENGTH * bs, emb_size]
		"""
		bs = images.shape[0]
		features = self.network(images.reshape(bs * images.shape[1], *images.shape[2:]))
		return features

	def uniqueness_loss(self):

		error = 0
		for i in range(self.num_states):
			distance = torch.matmul(self.embeddings[i], self.embeddings[i].t()) - torch.eye(self.num_objects).to(self.embeddings.device)
			norm = torch.norm(self.embeddings[i], dim=1, keepdim=True)
			normalisation = torch.matmul(norm, norm.t()) + 1e-4
			error += torch.mean(distance/normalisation ** 2)

		return error

	def similarity(self, features, tracklens):
		"""
		Input: [MAX_TRACK_LENGTH * bs, emb_size]
		Output: [num_states, num_objects, bs, MAX_TRACK_LEN]
		"""
		# compute similarities
		normalize = "normalize" in self.sim_type
		a = F.normalize(self.embeddings, p=2, dim=-1) if normalize else self.embeddings
		b = F.normalize(features, p=2, dim=-1) if normalize else features
		if "euclidean" in self.sim_type:
			sim = -torch.cdist(a.reshape(-1, self.emb_size), b)
			sim = sim.reshape(self.num_states, self.num_objects, features.shape[0])
		elif "cosine" in self.sim_type:
			sim = torch.matmul(a, b.t().unsqueeze(0))
		else:
			raise ValueError(f"distance type {self.sim_type} not implemented")
		# mask according to track length
		sim = sim.reshape(self.num_states, self.num_objects, len(tracklens), MAX_TRACK_LEN)
		positions = torch.arange(MAX_TRACK_LEN, device=sim.device)
		mask = positions.unsqueeze(0) >= tracklens.unsqueeze(1)
		sim[..., mask] = float('-inf')
		return sim

	def get_expected_position(self, sim, tracklens, temperature=None):
		"""
		Input: [num_states, num_objects, bs, MAX_TRACK_LEN], [bs]
		Output: [num_states, num_objects, bs]
		"""
		temperature = self.temperature if temperature is None else temperature
		positions = torch.arange(MAX_TRACK_LEN, device=tracklens.device)
		distribution = F.softmax(sim/temperature, dim=-1)
		return (distribution * positions).sum(dim=-1), distribution

	def forward(self, images, tracklens):
		"""
		Input: [bs, MAX_TRACK_LENGTH, 3, height, width], [bs]
		Output [num_states, num_objects, batch_size]
		"""
		features = self.get_features(images)
		sim = self.similarity(features, tracklens)
		return sim