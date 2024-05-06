
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import get_average_grad, get_best_candidates
import minigpt_v2_utils.prompt_wrapper as prompt_wrapper
import sys
sys.path.append('/data/home/wangyouze/projects/jailbreak_attack/MiniGPT-v2/')
from jailbreak_attack.visual_attack import VisualAttacker
from jailbreak_attack.text_attack import TextAttacker
import minigpt_v2_utils.generator as generator
from copy import deepcopy
import random
from tqdm import tqdm
from torchvision.utils import save_image
import json

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



class MultimodalStepsJailbreakAttack(object):
	def __init__(self, model, tokenizer, embedding_weight, conv_template, test_prefixes, iters, json_file_path, device):
		self.model = model
		self.tokenizer = tokenizer
		self.embedding_weight = embedding_weight

		self.conv_template = conv_template
		self.iters = iters
		self.test_prefixes = test_prefixes

		
		self.device = device
		self.save_dir = '/data/home/wangyouze/projects/jailbreak_attack/MiniGPT-v2/jailbreak_attack/results/adv_images/'
		self.json_file_path = json_file_path

	def _update_ids(self, goal, control, target):

		
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
			loss_slice = slice(self._loss_slice.start+padding, self._loss_slice.stop+padding)
			target_slice = slice(self._target_slice.start+padding, self._target_slice.stop+padding)
			if padding > 0:
				aligned_t = F.pad(t, (padding, 0), "constant", 0)
			else:
				aligned_t = t
			aligned_tensors.append(aligned_t.unsqueeze(0))
		
		res = torch.cat(aligned_tensors, dim=0)
		return res, control_slice, loss_slice, target_slice
	
	def attack(self, train_goals, test_goals, enhanced_goals, image, adv_control, target_label, batch_size=20,):

		# batch_goals_token_ids = self.tokenizer(batch_goals, padding=True, return_tensors=True)

		textual_attack = TextAttacker(self.model, self.tokenizer, goals=train_goals, targets=target_label, conv_template=self.conv_template, test_prefixes=self.test_prefixes, n_candidates=2, device=self.device)
		visual_attack = VisualAttacker(self.model, self.tokenizer, target_label, test_prefixes=self.test_prefixes, conv_template=self.conv_template, device=self.device)
		# vocabs, embedding_matrix = textual_attack.vocabs, textual_attack.embedding_matrix

		my_generator = generator.Generator(model=self.model)

		enhanced_goals_list = []
		
		if len(enhanced_goals) > 0:
			for k,values in enhanced_goals.items():
				for v in values:
					enhanced_goals_list.append(v)

		adv_control_tokens = self.tokenizer([adv_control], return_tensors="pt").input_ids[:, 1:].squeeze(0).to(self.model.device)
		if adv_control_tokens.shape[0] > 10:
			adv_control_tokens = adv_control_tokens[:10]

		for epoch in tqdm(range(1, self.iters+1)):
			print('epoch=', epoch)

			print('starting to perturb the image >>>>>')
			print('adv_control=', adv_control)
			print('randomly selected goals>>>>')
			print('[image batch_size]>>>', batch_size)
			batch_goals = random.sample(train_goals, batch_size)
			if len(enhanced_goals) > 0:
				batch_enhanced_goals = []
				for k, v in enhanced_goals.items():
					if k  in batch_goals:
						for x in v:
							batch_enhanced_goals.append(x)

			

				batch_enhanced_goals = random.sample(enhanced_goals_list, 35)

				batch_goals = train_goals + batch_enhanced_goals
			else:
				batch_goals = train_goals

			batch_goals_sample = random.sample(batch_goals, batch_size)

			img_prompts_list = []
			for i in range(batch_size):
				separator = ' '
				text_prompt = f"[INST] <Img><ImageHere></Img> {batch_goals_sample[i]+'.'}{separator}{adv_control} [/INST] "
				img_prompts_list.append(text_prompt)
			# '<s>[INST] <Img><ImageHere></Img> describe the image [/INST]'

			image = visual_attack.attack_vmifgsm(
												text_prompts=img_prompts_list, img=image, 
												batch_size=batch_size,
												num_iter=50, alpha=1./255)
				
			
			print('starting to perturb the text prompt>>>>>')

			
			v = torch.zeros((10, self.model.llama_model.base_model.model.model.embed_tokens.weight.shape[0])).detach().to(self.model.device)
			
			text_batch_size = batch_size
			# text_batch_goals_sample = random.sample(batch_goals, text_batch_size)
			print('[text batch_size]>>>', text_batch_size)
			for t in tqdm(range(0, 20)):

				print('[adv_control]>>>', adv_control)
		
				control_slice = []
				text_prompts_list = []

				
				for i in range(text_batch_size):


					# self.conv_template.append_message(self.conv_template.roles[0],  DEFAULT_IMAGE_TOKEN + '\n')
					# separator = ' '
					# self.conv_template.update_last_message(f"{DEFAULT_IMAGE_TOKEN}{batch_goals_sample[i]+'.'}{separator}{adv_control}")
					# self.conv_template.append_message(self.conv_template.roles[1], None)
					# # self.conv_template.update_last_message(f"{self.targets}")
					# text_prompt_suffix = self.conv_template.get_prompt()
					# # input_ids = self.tokenizer(prompt, return_tensors='pt').input_ids.to(self.model.device)
					# self.conv_template.messages = []
					
					separator = ' '
					text_prompt_suffix = f"[INST] <Img><ImageHere></Img> {batch_goals_sample[i]+'.'}{separator}{adv_control} [/INST] "

					text_prompts_list.append(text_prompt_suffix)
					
					input_ids = self._update_ids(batch_goals_sample[i]+'.', adv_control, target_label).to(self.model.device)
					# batch_input_ids.append(input_ids.squeeze(0))
					control_slice.append(self._control_slice)
					

				# adv_control, one_hot_momentum, loss  = textual_attack.attack(one_hot_momentum, control_slice, adv_control, text_prompts_list, image, batch_goals_sample, batch_size=text_batch_size, unwanted_token_index=unwanted_token_index)

				adv_control, adv_control_tokens, loss, v  = textual_attack.attack_vmifgsm(v, control_slice, adv_control, adv_control_tokens, text_prompts_list, image, batch_goals_sample, batch_size=text_batch_size)
			

				# torch.cuda.empty_cache()
			save_image(image.cpu().squeeze(0), '%s/adv_image_test_0429_%d.bmp' % (self.save_dir, epoch))
			# save_images_png(self.save_dir, image.cpu(), [str(epoch)+'.png'])
			torch.save(image.cpu().detach(), '%s/adv_image_test_0429_%d.pt' % (self.save_dir, epoch))
			if epoch % 10 == 0:

				
				### train goals acc
				train_acc, test_acc = [], []
				for t_goal in train_goals:
					# self.conv_template.append_message(self.conv_template.roles[0],  DEFAULT_IMAGE_TOKEN + '\n')
					separator = ' '
					# self.conv_template.update_last_message(f"{DEFAULT_IMAGE_TOKEN}{t_goal+'.'}{separator}{adv_control}")
					# self.conv_template.append_message(self.conv_template.roles[1], None)
					# # self.conv_template.update_last_message(f"{self.targets}")
					# text_prompts = self.conv_template.get_prompt()
					# # input_ids = self.tokenizer(prompt, return_tensors='pt').input_ids.to(self.model.device)
					# self.conv_template.messages = []

					# text_prompts = self.tokenizer.batch_decode(input_ids)

					text_prompts = f"'<s>[INST] <Img><ImageHere></Img> {t_goal+'.'}{separator}{adv_control} [/INST]"
				
					train_prompts = prompt_wrapper.Prompt(model=self.model, text_prompts=[text_prompts], device=self.device)

					train_prompts.update_img_prompts([[image]])
				
					train_prompts.update_context_embs()


					with torch.no_grad():
						response = my_generator.generate(train_prompts)
					jailbroken = not any([prefix in response for prefix in self.test_prefixes])
					train_acc.append(int(jailbroken))
				train_res = sum(train_acc)/len(train_acc)
				print('train_acc=', train_res)

				for test_goal in test_goals:
					# self.conv_template.append_message(self.conv_template.roles[0],  DEFAULT_IMAGE_TOKEN + '\n')
					separator = ' '
					# self.conv_template.update_last_message(f"{DEFAULT_IMAGE_TOKEN}{test_goal+'.'}{separator}{adv_control}")
					# self.conv_template.append_message(self.conv_template.roles[1], None)
					# # self.conv_template.update_last_message(f"{self.targets}")
					# text_prompts = self.conv_template.get_prompt()
					# # input_ids = self.tokenizer(prompt, return_tensors='pt').input_ids.to(self.model.device)
					# self.conv_template.messages = []

					# text_prompts = self.tokenizer.batch_decode(input_ids)

					text_prompts = f"'<s>[INST] <Img><ImageHere></Img> {test_goal+'.'}{separator}{adv_control} [/INST]"
				
					test_prompts = prompt_wrapper.Prompt(model=self.model,  text_prompts=[text_prompts], device=self.device)

					test_prompts.update_img_prompts([[image]])
					test_prompts.update_context_embs()


					with torch.no_grad():
						response = my_generator.generate(test_prompts)
						print('-----------------------------------------------------------------------')
						print('[Prompt]>>>', text_prompts)
						print('[response]>>>', response)
					jailbroken = not any([prefix in response for prefix in self.test_prefixes])

					if jailbroken:
						print('successfully!!!')

					test_acc.append(int(jailbroken))
				test_res = sum(test_acc)/len(test_acc)
				print('test_acc=', test_res)

				experiment_results = [{"epoch": epoch, "adv_control": adv_control, "train_acc": train_res,"test_acc": test_res}]
				
				with open(self.json_file_path, "a") as f:
					for result in experiment_results:
						json.dump(result, f) 
						f.write('\n')  
			else:

				experiment_results = [{"epoch": epoch, "adv_control": adv_control}]
				
				with open(self.json_file_path, "a") as f:
					for result in experiment_results:
						json.dump(result, f) 
						f.write('\n') 
		return adv_control, image





	