"""CPU checks for the released main method; no pretrained weights needed."""
import unittest
import torch
from skillflow.flow.model import (VelocityField, sample_meanflow_endpoint,
    improved_meanflow_boundary_target)
from skillflow.io import load_config, training_rows
from skillflow.train_flow import hierarchy_weights, split_rows
from skillflow.generate import public_conditions


class CoreTest(unittest.TestCase):
    def test_interval_average_one_field_call(self):
        class Field(torch.nn.Module):
            calls = 0
            def forward(self, x, r, t, q):
                self.calls += 1
                torch.testing.assert_close(r, torch.zeros_like(r))
                torch.testing.assert_close(t, torch.ones_like(t))
                torch.testing.assert_close(q.norm(dim=-1), torch.ones_like(t))
                # The diagonal r=t yields zero, unlike the interval r=0, t=1.
                direction = x.new_tensor([0.5, -0.25])
                return (t - r).unsqueeze(-1) * direction
        field = Field()
        x = torch.tensor([[2.0, 1.0], [1.0, -2.0]])
        q = torch.tensor([[3.0, 4.0], [-4.0, 3.0]])
        result = sample_meanflow_endpoint(field, x, q)
        self.assertEqual(field.calls, 1)
        expected = torch.nn.functional.normalize(x - x.new_tensor([0.5, -0.25]), dim=-1)
        torch.testing.assert_close(result, expected)
        self.assertFalse(torch.allclose(result, torch.nn.functional.normalize(x, dim=-1)))
        for n in (0, 2, True, 1.0):
            with self.assertRaises(ValueError): sample_meanflow_endpoint(field, x, q, sampling_steps=n)
        self.assertEqual(field.calls, 1)

    def test_boundary_objective_has_finite_gradients(self):
        torch.manual_seed(1)
        model = VelocityField(z_dim=16,q_dim=8,hidden_dim=32,depth=1,time_dim=16,
                              context_dim=32,attention_heads=4,latent_token_size=8)
        x,q = torch.randn(4,16),torch.randn(4,8)
        r,t = torch.full((4,),0.2),torch.full((4,),0.7)
        target,_,jvp = improved_meanflow_boundary_target(model,x,r,t,q,conditional_velocity=torch.randn_like(x))
        self.assertFalse(target.requires_grad)
        self.assertTrue(torch.isfinite(jvp).all())
        (model(x,r,t,q)-target).square().sum(-1).mean().backward()
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_training_data_and_holdout(self):
        for name,n in [('alfworld',3553),('nq',4000),('multihop',2723),('webshop',2000)]:
            cfg=load_config('configs/'+name+'.yaml');rows=training_rows(cfg)
            self.assertEqual(len(rows),n)
            self.assertTrue(all(isinstance(r['executable_skill'],dict) for r in rows))
            train,probe,independent=split_rows(rows,cfg['flow'])
            self.assertTrue(probe)
            if independent: self.assertFalse(set(train)&set(probe))
            else: self.assertEqual(len(train),n)
            self.assertAlmostEqual(float(hierarchy_weights([rows[i] for i in train]).sum()),1.0)

    def test_only_public_conditions(self):
        rows=public_conditions([{'task_id':'a','question':'Who wrote this book?',
                                 'golden_answers':['SECRET'],'explanation':'SECRET'}],'nq')
        self.assertNotIn('SECRET',str(rows))
        self.assertEqual(set(rows[0]),{'task_id','task_type','canonical_goal'})
        row=public_conditions([{'task_id':'b','goal':'Find running shoes'}],'webshop')[0]
        self.assertEqual(row['task_type'],'footwear')

if __name__=='__main__': unittest.main()
