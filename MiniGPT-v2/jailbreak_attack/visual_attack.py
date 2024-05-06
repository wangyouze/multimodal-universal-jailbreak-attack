import torch
import torch.nn as nn
from tqdm import tqdm
import random

import minigpt_v2_utils.prompt_wrapper as prompt_wrapper
import  minigpt_v2_utils.generator as generator
from torchvision.utils import save_image

import matplotlib.pyplot as plt
import seaborn as sns
DEFAULT_IMAGE_TOKEN='<Img><ImageHere></Img>'


def normalize(images):
	mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).cuda()
	std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).cuda()
	images = images - mean[None, :, None, None]
	images = images / std[None, :, None, None]
	return images

def denormalize(images):
	mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).cuda()
	std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).cuda()
	images = images * std[None, :, None, None]
	images = images + mean[None, :, None, None]
	return images


class VisualAttacker:

	def __init__(self,  model, tokenizer, targets, test_prefixes, conv_template, device='cuda', is_rtp=False):

		# self.args = args
		self.model = model
		self.tokenizer = tokenizer
		self.device = device
		self.is_rtp = is_rtp

		self.targets = targets
		self.num_targets = len(targets)

		self.loss_buffer = []

		# freeze and set to eval model:
		self.model.eval()
		self.model.requires_grad_(False)

		self.test_prefixes = test_prefixes
		self.conv_template = conv_template

	def attack_unconstrained(self, text_prompt, img, batch_size = 8, num_iter=2000, alpha=1/255):

		print('>>> batch_size:', batch_size)

		my_generator = generator.Generator(model=self.model)

		adv_noise = torch.rand_like(img).to(self.device) # [0,1]
		adv_noise.requires_grad_(True)
		adv_noise.retain_grad()

		for t in tqdm(range(num_iter + 1)):

			batch_targets = random.sample(self.targets, batch_size)
			text_prompts = [text_prompt] * batch_size


			x_adv = normalize(adv_noise)

			prompt = prompt_wrapper.Prompt(model=self.model, text_prompts=text_prompts, img_prompts=[[x_adv]])
			prompt.img_embs = prompt.img_embs * batch_size
			prompt.update_context_embs()

			target_loss = self.attack_loss(prompt, batch_targets)
			target_loss.backward()

			adv_noise.data = (adv_noise.data - alpha * adv_noise.grad.detach().sign()).clamp(0, 1)
			adv_noise.grad.zero_()
			self.model.zero_grad()

			self.loss_buffer.append(target_loss.item())

			print("target_loss: %f" % (
				target_loss.item())
				  )

			if t % 20 == 0:
				self.plot_loss()

			if t % 100 == 0:
				print('######### Output - Iter = %d ##########' % t)
				x_adv = normalize(adv_noise)
				prompt.update_img_prompts([[x_adv]])
				prompt.img_embs = prompt.img_embs * batch_size
				prompt.update_context_embs()
				with torch.no_grad():
					response, _ = my_generator.generate(prompt)
				print('>>>', response)

				adv_img_prompt = denormalize(x_adv).detach().cpu()
				adv_img_prompt = adv_img_prompt.squeeze(0)
				# save_image(adv_img_prompt, '%s/bad_prompt_temp_%d.bmp' % (self.args.save_dir, t))

		return adv_img_prompt
	

	def attack_unconstrained_2(self, epoch, text_prompts, img, batch_size = 8, num_iter=2000, alpha=1/255, decay=1.0 ):

		print('>>> batch_size:', batch_size)
		non_targeted_text = "The image features a large panda bear with its mouth wide open, appearing to be growling or roaring. The panda bear is the main focus of the image, occupying a significant portion of the frame. The image is a close-up of the panda bear, allowing the viewer to appreciate its features and details. The panda bear's mouth is open, and its teeth are visible, adding to the intensity of the scene. Overall, the image captures the powerful and intimidating presence of the panda bear."
		
		if epoch < 1:
			# for normalized imgs
			scaling_tensor = torch.tensor((0.26862954, 0.26130258, 0.27577711), device=self.device)
			scaling_tensor = scaling_tensor.reshape((3, 1, 1)).unsqueeze(0)

			alpha = alpha/scaling_tensor

			adv_noise = torch.rand_like(img).to(self.device) # [0,1]
			adv_noise.requires_grad_(True)
			adv_noise.retain_grad()
		else:
			
			adv_noise = img.clone()
			adv_noise.requires_grad_(True)
			adv_noise.retain_grad()
		
		prompt = prompt_wrapper.Prompt(self.model, text_prompts=text_prompts, device=self.device)
		momentum = torch.zeros_like(img).detach().to(self.device)

		response_list = []
		for t in range(1, num_iter + 1):
			
			response_list.append(non_targeted_text)

			batch_targets = [self.targets] * batch_size

			prompt.update_img_prompts([[adv_noise]])
			prompt.img_embs = prompt.img_embs * batch_size
			prompt.update_context_embs()
			

			refused_text = random.sample(response_list, 1)
			target_loss = self.attack_loss(prompt, batch_targets, refused_text[0])
			target_loss.backward()
			response_list = []

			grad = adv_noise.grad + momentum * decay
			momentum = grad
	
			adv_noise.data = (adv_noise.data - alpha * grad.detach().sign()).clamp(0, 1)
			adv_noise.grad.zero_()
			self.model.zero_grad()
			

			print("target_loss: %f" % (
				target_loss.item())
				  )

			# if t % 20 == 0:
			#     self.plot_loss()
			
		adv_img_prompt = adv_noise.detach()
	
		return adv_img_prompt

	


	def attack_vmifgsm(self, text_prompts, img, batch_size = 8, num_iter=2000, alpha=1/255,  decay=1.0, unwanted_token_index=None):
		eps = 32./255
		beta=3/2
		img = img.clone().detach().to(self.device)
		momentum = torch.zeros_like(img).detach().to(self.device)
		v = torch.zeros_like(img).detach().to(self.device)

		adv_img = img.clone().detach()

		prompt = prompt_wrapper.Prompt(self.model, text_prompts=text_prompts, device=self.device)
		batch_targets = [self.targets] * batch_size
		for _ in range(num_iter):
			adv_img.requires_grad = True

			prompt.update_img_prompts([[adv_img]])
			prompt.img_embs = prompt.img_embs * batch_size
			prompt.update_context_embs()

			loss = -self.attack_loss(prompt, batch_targets)
			print("target_loss: %f" % (
				loss.item())
				  )

			adv_grad = torch.autograd.grad(
				loss, adv_img, retain_graph=False, create_graph=False
			)[0]

			grad = (adv_grad + v) / torch.mean(
				torch.abs(adv_grad + v), dim=(1, 2, 3), keepdim=True
			)
			grad = grad + momentum * decay
			momentum = grad

			# Calculate Gradient Variance
			GV_grad = torch.zeros_like(img).detach().to(self.device)
			for _ in range(5):
				neighbor_images = adv_img.detach() + torch.randn_like(
					img
				).uniform_(-eps * beta, eps * beta)
				neighbor_images.requires_grad = True

				prompt.update_img_prompts([[neighbor_images]])
				prompt.img_embs = prompt.img_embs * batch_size
				prompt.update_context_embs()

				loss = -self.attack_loss(prompt, batch_targets)

				GV_grad += torch.autograd.grad(
					loss, neighbor_images, retain_graph=False, create_graph=False
				)[0]
			# obtaining the gradient variance
			v = GV_grad / 5 - adv_grad
			adv_img = adv_img.detach() + alpha * grad.sign()
			delta = torch.clamp(adv_img - img, min=-eps, max=eps)
			adv_img = torch.clamp(img + delta, min=0, max=1).detach()
		return adv_img

	


	def attack_loss(self, prompts, targets, non_targeted_text=None):

		# non_targeted_text = "The image features a large panda bear with its mouth wide open, appearing to be growling or roaring. The panda bear is the main focus of the image, occupying a significant portion of the frame. The image is a close-up of the panda bear, allowing the viewer to appreciate its features and details. The panda bear's mouth is open, and its teeth are visible, adding to the intensity of the scene. Overall, the image captures the powerful and intimidating presence of the panda bear."
		# non_targeted_tokens_ids = self.tokenizer([non_targeted_text] * len(targets), return_tensors='pt')['input_ids'].to(self.device)


		context_embs = prompts.context_embs

		if len(context_embs) == 1:
			context_embs = context_embs * len(targets) # expand to fit the batch_size

		assert len(context_embs) == len(targets), f"Unmathced batch size of prompts and targets {len(context_embs)} != {len(targets)}"

		batch_size = len(targets)
		self.model.llama_tokenizer.padding_side = "right"

		to_regress_tokens = self.model.llama_tokenizer(
			targets,
			return_tensors="pt",
			padding="longest",
			truncation=True,
			max_length=self.model.max_txt_len,
			add_special_tokens=False
		).to(self.device)
		to_regress_embs =self.model.llama_model.base_model.model.model.embed_tokens(to_regress_tokens.input_ids)

		bos = torch.ones([1, 1],
						 dtype=to_regress_tokens.input_ids.dtype,
						 device=to_regress_tokens.input_ids.device) * self.model.llama_tokenizer.bos_token_id
		bos_embs = self.model.llama_model.base_model.model.model.embed_tokens(bos)

		pad = torch.ones([1, 1],
						 dtype=to_regress_tokens.input_ids.dtype,
						 device=to_regress_tokens.input_ids.device) * self.model.llama_tokenizer.pad_token_id
		pad_embs = self.model.llama_model.base_model.model.model.embed_tokens(pad)


		T = to_regress_tokens.input_ids.masked_fill(
			to_regress_tokens.input_ids == self.model.llama_tokenizer.pad_token_id, -100
		)


		pos_padding = torch.argmin(T, dim=1) # a simple trick to find the start position of padding

		input_embs = []
		targets_mask = []

		target_tokens_length = []
		context_tokens_length = []
		seq_tokens_length = []

		for i in range(batch_size):

			pos = int(pos_padding[i])
			if T[i][pos] == -100:
				target_length = pos
			else:
				target_length = T.shape[1]

			targets_mask.append(T[i:i+1, :target_length])
			input_embs.append(to_regress_embs[i:i+1, :target_length]) # omit the padding tokens

			context_length = context_embs[i].shape[1]
			seq_length = target_length + context_length

			target_tokens_length.append(target_length)
			context_tokens_length.append(context_length)
			seq_tokens_length.append(seq_length)

		max_length = max(seq_tokens_length)

		attention_mask = []

		for i in range(batch_size):

			# masked out the context from loss computation
			context_mask =(
				torch.ones([1, context_tokens_length[i] + 1],
					   dtype=torch.long).to(self.device).fill_(-100)  # plus one for bos
			)

			# padding to align the length
			num_to_pad = max_length - seq_tokens_length[i]
			padding_mask = (
				torch.ones([1, num_to_pad],
					   dtype=torch.long).to(self.device).fill_(-100)
			)

			targets_mask[i] = torch.cat( [ padding_mask, context_mask, targets_mask[i]], dim=1 )
			input_embs[i] = torch.cat( [pad_embs.repeat(1, num_to_pad, 1), bos_embs, context_embs[i], input_embs[i],
										], dim=1 )
			attention_mask.append( torch.LongTensor( [[0]*num_to_pad + [1]* (1+seq_tokens_length[i]) ] ) )

		targets = torch.cat( targets_mask, dim=0 ).to(self.device)
		inputs_embs = torch.cat( input_embs, dim=0 ).to(self.device)
		attention_mask = torch.cat(attention_mask, dim=0).to(self.device)


		outputs = self.model.llama_model(
				inputs_embeds=inputs_embs,
				attention_mask=attention_mask,
				return_dict=True,
				labels=targets,
			)
		loss = outputs.loss


		# criterion = nn.CrossEntropyLoss()  # 忽略填充值对损失的影响

		# logits = outputs.logits
		# target_logits = logits[:, -to_regress_embs.shape[1]:, :]
		# if target_logits.shape[1] > non_targeted_tokens_ids.shape[1]:
		# 	lens = non_targeted_tokens_ids.shape[1]
		# 	target_logits = target_logits[:, :lens, :]
		# else:
		# 	lens = target_logits.shape[1]
		# 	non_targeted_tokens_ids = non_targeted_tokens_ids[:, :lens]
		# target_logits = target_logits.reshape(-1, target_logits.size(-1))
		# non_targeted_tokens_ids = non_targeted_tokens_ids.reshape(-1)
		# loss_2 = criterion(target_logits, non_targeted_tokens_ids)

		
		return 10 *loss 
	
	def target_loss(self, logits, ids):
		crit = nn.CrossEntropyLoss(reduction='none')
		loss_slice = slice(self._target_slice.start-1, self._target_slice.stop-1)
		loss = crit(logits[:,loss_slice,:].transpose(1,2), ids[:,self._target_slice])
		return loss