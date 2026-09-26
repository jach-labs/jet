"""Task-weighted supervision plus forward KL against released Jet on train rows only."""
import torch
import torch.nn.functional as F
from qwen35_training import decision_loss

def teacher_kl(logits,teacher,temperature=2.):
 target=torch.as_tensor(teacher,device=logits.device,dtype=logits.dtype).detach()
 return F.kl_div(F.log_softmax(logits/temperature,dim=-1),F.softmax(target/temperature,dim=-1),reduction='sum')*temperature**2

def anchored_loss(logits,target,ordinal,family,teacher,strength):
 weight={'sarcasm':4.,'deadline':3.}.get(family,1.)
 anchor_weight=.25 if family in ['and_or','or_not','conditional'] else 1.
 return decision_loss(logits,target,ordinal,weight)+strength*anchor_weight*teacher_kl(logits,teacher)
