import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
sys.path.insert(0,str(Path(__file__).resolve().parent))
import torch
from objective import teacher_kl,anchored_loss
from train import eligible,GUARDS
class ObjectiveTests(unittest.TestCase):
 def test_matching_teacher_has_zero_gradient(self):
  z=torch.tensor([1.,-2.,3.],requires_grad=True);loss=teacher_kl(z,z.detach());loss.backward()
  self.assertAlmostEqual(loss.item(),0.,places=6);self.assertLess(z.grad.abs().max().item(),1e-6)
 def test_gradient_moves_distribution_toward_teacher(self):
  z=torch.tensor([-1.,1.],requires_grad=True);teacher=torch.tensor([1.,-1.],requires_grad=True);before=teacher_kl(z,teacher);before.backward()
  self.assertIsNone(teacher.grad);self.assertLess(teacher_kl(z-.1*z.grad,teacher).item(),before.item())
 def test_all_guards_preserved(self):
  base={'families':{k:{'metric':.8} for k in GUARDS}}
  for key in GUARDS:
   candidate={'families':{k:{'metric':.8 if k!=key else .77} for k in GUARDS}}
   self.assertFalse(eligible(candidate,base),key)
 def test_finite_task_losses(self):
  for family in GUARDS:
   z=torch.tensor([0.,1.,-1.],requires_grad=True)
   loss=anchored_loss(z,torch.tensor([0.,1.,0.]),family=='deadline',family,[1.,0.,-1.],2.)
   loss.backward();self.assertTrue(torch.isfinite(z.grad).all())
if __name__=='__main__':unittest.main()
