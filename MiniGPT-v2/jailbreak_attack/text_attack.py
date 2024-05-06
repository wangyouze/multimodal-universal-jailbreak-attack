import torch
import torch.nn as nn
from tqdm import tqdm
import random
import sys
from torchvision.utils import save_image
import numpy as np
from copy import deepcopy
import time

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import MultiCursor
import seaborn as sns
import torch.nn.functional as F
import minigpt_v2_utils.prompt_wrapper as prompt_wrapper

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

class TextAttacker:

	def __init__(self, model, tokenizer, goals, targets, conv_template, test_prefixes, n_candidates=20, device='cuda'):

		# self.args = args
		self.model = model
		self.tokenizer = tokenizer
		self.device = device

		self.goals = goals
		self.targets = targets # targets that we want to promte likelihood
		self.loss_buffer = []
		self.num_targets = len(self.targets)

		# freeze and set to eval model:
		self.model.eval()
		self.model.requires_grad_(False)
		self.tokenizer.padding_side = "right"
		self.n_candidates = n_candidates
		self.conv_template = conv_template

		self.test_prefixes = test_prefixes

		self.get_vocabulary()

	def get_vocabulary(self):

		vocab_dicts = self.tokenizer.get_vocab()
		vocabs = vocab_dicts.keys()

		single_token_vocabs = []
		single_token_vocabs_embedding = []
		single_token_id_to_vocab = dict()
		single_token_vocab_to_id = dict()

		cnt = 0

		for item in vocabs:
			tokens = self.tokenizer(item, return_tensors="pt", add_special_tokens=False).input_ids.to(self.model.device)
			# if tokens.shape[1] == 1:

			single_token_vocabs.append(item)
			emb = self.model.llama_model.base_model.model.model.embed_tokens(tokens)
			single_token_vocabs_embedding.append(emb)

			single_token_id_to_vocab[cnt] = item
			single_token_vocab_to_id[item] = cnt

			cnt+=1

		single_token_vocabs_embedding = torch.cat(single_token_vocabs_embedding, dim=1).squeeze()

		self.vocabs = single_token_vocabs
		self.embedding_matrix = single_token_vocabs_embedding.to(self.device)
		self.id_to_vocab = single_token_id_to_vocab
		self.vocab_to_id = single_token_vocab_to_id

	def _update_ids(self, goal, control, target):

		self.conv_template.append_message(self.conv_template.roles[0], f"{goal} {control}")
		self.conv_template.append_message(self.conv_template.roles[1], f"{target}")
		prompt = self.conv_template.get_prompt()
		toks = self.tokenizer(prompt, return_tensors='pt')

		
		self.conv_template.messages = []

		self.conv_template.append_message(self.conv_template.roles[0],  DEFAULT_IMAGE_TOKEN + '\n')
		toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
		self._user_role_slice = slice(None, len(toks)-2)

		self.conv_template.update_last_message(f"{DEFAULT_IMAGE_TOKEN}{goal}")
		toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
		self._goal_slice = slice(self._user_role_slice.stop, max(self._user_role_slice.stop, len(toks)-1))

		separator = ' ' if goal else ''
		self.conv_template.update_last_message(f"{DEFAULT_IMAGE_TOKEN}{goal}{separator}{control}")
		toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
		self._control_slice = slice(self._goal_slice.stop, len(toks)-1)
		if self._control_slice.stop - self._control_slice.start < 10:
			self._control_slice = slice(self._control_slice.start- (10 - (self._control_slice.stop - self._control_slice.start) ), self._control_slice.stop)
		if self._control_slice.stop - self._control_slice.start > 10:
			self._control_slice = slice(self._control_slice.start+ ((self._control_slice.stop - self._control_slice.start) - 10 ), self._control_slice.stop)
		

		self.conv_template.append_message(self.conv_template.roles[1], None)
		toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
		self._assistant_role_slice = slice(self._control_slice.stop, len(toks))

		self.conv_template.update_last_message(f"{target}")
		toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
		self._target_slice = slice(self._assistant_role_slice.stop, len(toks)-1)
		self._loss_slice = slice(self._assistant_role_slice.stop-1, len(toks)-2)
		
		# self.input_ids = torch.tensor(toks[:self._target_slice.stop], device='cpu')
		self.input_ids = torch.tensor(toks, device='cpu')
		self.conv_template.messages = []
		return self.input_ids
	
	def target_loss(self, logits, ids):
		# crit = nn.CrossEntropyLoss(reduction='none')
		# loss = crit(logits.transpose(1,2), ids)

		loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), ids.reshape(-1), reduction='mean')

		return loss
	
	def get_cands(self, control_toks, grad, topk, batch_size):

		top_indices = (-grad).topk(topk, dim=1).indices
		control_toks = control_toks.to(grad.device)
		original_control_toks = control_toks.repeat(batch_size, 1)
		new_token_pos = torch.arange(0, len(control_toks), len(control_toks) / batch_size, device=grad.device).type(torch.int64)
		
		new_token_val = torch.gather(
			top_indices[new_token_pos], 1, 
			torch.randint(0, topk, (batch_size, 1),
			device=grad.device)
		)
		new_control_toks = original_control_toks.scatter_(1, new_token_pos.unsqueeze(-1), new_token_val)

		return new_control_toks
	def get_filtered_cands(self, control_cand, filter_cand=True, curr_control=None):
		cands, count = [], 0
		for i in range(control_cand.shape[0]):
			decoded_str = self.tokenizer.decode(control_cand[i], skip_special_tokens=True)
			if filter_cand:
				if decoded_str != curr_control and len(self.tokenizer(decoded_str, add_special_tokens=False).input_ids) == len(control_cand[i]):
					cands.append(decoded_str)
				else:
					count += 1
			else:
				cands.append(decoded_str)
				
		if filter_cand:
			cands = cands + [cands[-1]] * (len(control_cand) - len(cands))
			# print(f"Warning: {round(count / len(control_cand), 2)} control candidates were not valid")
		return cands
	

	def update_adv_prompt(self, adv_prompt_tokens, idx, new_token):
		next_adv_prompt_tokens = deepcopy(adv_prompt_tokens)
		next_adv_prompt_tokens[idx] = new_token
		next_adv_prompt = ' '.join(next_adv_prompt_tokens)
		return next_adv_prompt_tokens, next_adv_prompt

	def align_tensors(self, tensors_list):
		max_width = max(t.size(0) for t in tensors_list)

		aligned_tensors = []
		for t in tensors_list:
			padding = max_width - t.size(0)
			control_slice = slice(self._control_slice.start+padding, self._control_slice.stop+padding)
			if padding > 0:
				aligned_t = F.pad(t, (padding, 0), "constant", 0)
			else:
				aligned_t = t
			aligned_tensors.append(aligned_t.unsqueeze(0))
		
		res = torch.cat(aligned_tensors, dim=0)
		return res, control_slice
	
	def get_nonascii_toks(self, device):

		def is_ascii( s):
			return s.isascii() and s.isprintable()

		ascii_toks = []
		for i in range(3, self.tokenizer.vocab_size):
			if not is_ascii(self.tokenizer.decode([i])):
				ascii_toks.append(i)
		
		if self.tokenizer.bos_token_id is not None:
			ascii_toks.append(self.tokenizer.bos_token_id)
		if self.tokenizer.eos_token_id is not None:
			ascii_toks.append(self.tokenizer.eos_token_id)
		if self.tokenizer.pad_token_id is not None:
			ascii_toks.append(self.tokenizer.pad_token_id)
		if self.tokenizer.unk_token_id is not None:
			ascii_toks.append(self.tokenizer.unk_token_id)
		
		return torch.tensor(ascii_toks, device=device)




	def attack(self, one_hot_momentum, control_slice, adv_control, text_prompts, image, batch_goals, batch_size=8, decay=1.0):
		# image = normalize(image)
		adv_control_tokens = self.tokenizer([adv_control], return_tensors="pt").input_ids[:, 1:].squeeze(0).to(self.model.device)
		# print('trigger_token_length=', control_tensor_tokens.shape[0])

	

		batch_targets = [self.targets] * batch_size


		embed_weights = self.model.llama_model.base_model.model.model.embed_tokens.weight
		
		one_hot = torch.zeros(
			adv_control_tokens.shape[0],
			embed_weights.shape[0],
			device=self.model.device,
			dtype=embed_weights.dtype
		)
		one_hot.scatter_(
			1, 
			adv_control_tokens.unsqueeze(1),
			torch.ones(one_hot.shape[0], 1, device=self.model.device, dtype=embed_weights.dtype)
		)
		one_hot.requires_grad_()
		control_embeds = (one_hot @ embed_weights).unsqueeze(0)

		prompts = prompt_wrapper.Prompt(model=self.model, text_prompts=text_prompts, control_slice_list=control_slice, control_embeds=control_embeds, device=self.device)
		
		prompts.update_img_prompts([[image]])
		prompts.img_embs = prompts.img_embs * batch_size
		prompts.update_context_embs()
		
		loss = self.attack_loss(prompts, batch_targets)

		loss.backward()
		one_hot_grad = one_hot.grad
		one_hot_grad = one_hot_grad / torch.mean(torch.abs(one_hot_grad), dim=(0, 1), keepdim=True)
		one_hot_grad = one_hot_grad + one_hot_momentum * decay

		self.model.zero_grad()
		
		one_hot_momentum = one_hot_grad.clone()
		print("loss: %f" % (loss.item()))
		
		one_hot_grad[:, self.get_nonascii_toks(one_hot_grad.device)] = np.infty

		candidates = self.get_cands(adv_control_tokens, one_hot_grad, topk=128, batch_size=30)
		control_cands = self.get_filtered_cands(candidates, filter_cand=True, curr_control=adv_control)
		# try all the candidates and pick the best
		# comparing candidates does not require gradient computation

		# Search
		res = []
		loss_list = []
		separator = ' '
		with torch.no_grad():
			for index in range(len(candidates)):
				cand = candidates[index]
				control = self.tokenizer.decode(cand)
				text_prompts_list = []
				for i in range(batch_size):

					text_prompt =  f"<s>[INST] <Img><ImageHere></Img>{batch_goals[i]+'.'}{separator}{control} [/INST]:{separator}{self.targets}"
					
					text_prompts_list.append(text_prompt)
				
				prompts = prompt_wrapper.Prompt(model=self.model, text_prompts=text_prompts_list, device=self.device)
				prompts.update_img_prompts([[image]])
				prompts.img_embs = prompts.img_embs * batch_size
				prompts.update_context_embs()

				loss = self.attack_loss(prompts, batch_targets)
				loss_list.append(loss.item())
				res.append(loss)
				torch.cuda.empty_cache()

		res = torch.stack(res)
		min_idx = res.argmin()
		next_control, cand_loss = control_cands[min_idx.item()], loss_list[min_idx.item()]
		if cand_loss < loss:
			new_control_tokens = self.tokenizer(next_control).input_ids[1:]
			print('[Current length]>>>', len(new_control_tokens))
		else:
			next_control = adv_control
		# print('[next_control]>>>',next_control)
		return next_control, one_hot_momentum, loss


	def attack_vmifgsm(self, v, control_slice, adv_control, adv_control_tokens, text_prompts, image, batch_goals, batch_size=8, inner_iter=3, unwanted_token_index=None):
		
		momentum = 1.0
		decay = 1.0
		# adv_control_tokens = self.tokenizer([adv_control], return_tensors="pt").input_ids[:, 1:].squeeze(0).to(self.model.device)

		# prompts = prompt_wrapper.Prompt(model=self.model, text_prompts=text_prompts, device=self.device)

		
		batch_targets = [self.targets] * batch_size


		embed_weights =self.model.llama_model.base_model.model.model.embed_tokens.weight
		
		one_hot = torch.zeros(
			adv_control_tokens.shape[0],
			embed_weights.shape[0],
			device=self.model.device,
			dtype=embed_weights.dtype
		)
		one_hot.scatter_(
			1, 
			adv_control_tokens.unsqueeze(1),
			torch.ones(one_hot.shape[0], 1, device=self.model.device, dtype=embed_weights.dtype)
		)
		one_hot.requires_grad_()

		control_embeds = (one_hot @ embed_weights).unsqueeze(0)
		
		# control_embeds = control_embeds.repeat(batch_size, 1, 1)

		prompts = prompt_wrapper.Prompt(model=self.model, text_prompts=text_prompts, control_slice_list=control_slice, control_embeds=control_embeds, device=self.device)
		prompts.update_img_prompts([[image]])
		prompts.img_embs = prompts.img_embs * batch_size
		prompts.update_context_embs()


		current_loss = - self.attack_loss(prompts, batch_targets)

		print("target_loss: %f" % (
				current_loss.item())
				  )

		# current_loss.backward()
		adv_grad = torch.autograd.grad(
				current_loss, one_hot, retain_graph=False, create_graph=False
			)[0]

		grad =  (adv_grad + v) / torch.mean(torch.abs(adv_grad + v), dim=(0, 1), keepdim=True)

		grad = grad + momentum * decay
		momentum = grad

		# Calculate Gradient Variance
		GV_grad = torch.zeros_like(one_hot).detach().to(self.model.device)

		for _ in range(inner_iter):
			neighbor_one_hot = one_hot.detach() + torch.randn_like(
				one_hot
			).uniform_(-1, 1)
			neighbor_one_hot.requires_grad = True

			ner_control_embeds = (neighbor_one_hot @ embed_weights).unsqueeze(0)
		
			# ner_control_embeds = ner_control_embeds.repeat(batch_size, 1, 1)

			prompts = prompt_wrapper.Prompt(model=self.model, text_prompts=text_prompts, control_slice_list=control_slice, control_embeds=ner_control_embeds, device=self.device)
			prompts.update_img_prompts([[image]])
			prompts.img_embs = prompts.img_embs * batch_size
			prompts.update_context_embs()

			loss = -self.attack_loss(prompts,  batch_targets)

			GV_grad += torch.autograd.grad(
				loss, neighbor_one_hot, retain_graph=False, create_graph=False
			)[0]
		# obtaining the gradient variance
		v = GV_grad /inner_iter - adv_grad
		
		grad[:, self.get_nonascii_toks(grad.device)] = np.infty

		candidates = self.get_cands(adv_control_tokens, grad, topk=128, batch_size=256)
		# control_cands = self.get_filtered_cands(candidates, filter_cand=True, curr_control=adv_control)
		# try all the candidates and pick the best
		# comparing candidates does not require gradient computation

		# Search
		res = []
		loss_list = []
		separator = ' '

		# img_embeds = self.model.encode_images(image).mean(dim=1)
		# candidates_embeds = embed_weights[candidates].mean(dim=1).to(img_embeds.device)
		# sim =  torch.norm(candidates_embeds - img_embeds, dim=1)
		# sim_index = torch.topk(-sim, 30, largest=False)[1].tolist()

		control_cands = []
		control_cands_tokens = []
		with torch.no_grad():
			for index in range(len(candidates)):
				# if index not in sim_index:
				# 	continue
				cand = candidates[index]
				if cand.shape[0] > 10:
					cand = cand[:10]
				control = self.tokenizer.decode(cand)
				text_prompts_list = []
				for i in range(batch_size):
					
					# text_prompt = f"{self.conv_template.system} USER: <image>{batch_goals[i]+'.'}{separator}{control} ASSISTANT:{separator}{self.targets}"
					text_prompt = f"'<s>[INST] <Img><ImageHere></Img> {batch_goals[i]+'.'}{separator}{control} [/INST] "
					
					text_prompts_list.append(text_prompt)
				
				prompts = prompt_wrapper.Prompt(model=self.model, text_prompts=text_prompts_list, device=self.device)

				prompts.update_img_prompts([[image]])
				prompts.img_embs = prompts.img_embs * batch_size
				prompts.update_context_embs()

				loss = self.attack_loss(prompts, batch_targets)
				loss_list.append(loss.item())
				res.append(loss.abs())
				control_cands.append(control)
				control_cands_tokens.append(cand)
				torch.cuda.empty_cache()

		res = torch.stack(res)
		min_idx = res.argmin()
		next_control, cand_loss = control_cands[min_idx.item()], loss_list[min_idx.item()]
		next_control_token = control_cands_tokens[min_idx.item()]
		if np.abs(cand_loss) < torch.abs(current_loss):
			new_control_tokens = self.tokenizer(next_control).input_ids[1:]
			print('[Current length]>>>', len(new_control_tokens))
		else:
			next_control = adv_control
			next_control_token = adv_control_tokens
		# print('[next_control]>>>',next_control)
		return next_control, next_control_token, loss, v


	
	def attack_loss(self, prompts, targets):

		non_targeted_text = "The image features a large panda bear with its mouth wide open, appearing to be growling or roaring. The panda bear is the main focus of the image, occupying a significant portion of the frame. The image is a close-up of the panda bear, allowing the viewer to appreciate its features and details. The panda bear's mouth is open, and its teeth are visible, adding to the intensity of the scene. Overall, the image captures the powerful and intimidating presence of the panda bear."
		non_targeted_tokens_ids = self.tokenizer([non_targeted_text] * len(targets), return_tensors='pt')['input_ids'].to(self.device)


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
		# loss = outputs.loss


		criterion = nn.CrossEntropyLoss()  # 忽略填充值对损失的影响

		logits = outputs.logits
		target_logits = logits[:, -to_regress_tokens.input_ids.shape[1]:, :]
		# target_logits = target_logits.reshape(-1, target_logits.size(-1))
		losses = []
		for i in range(target_logits.shape[0]):
			losses.append(criterion(target_logits[i], to_regress_tokens.input_ids[i]))
		loss = torch.stack(losses).mean(dim=0)

		return 10 *loss
	