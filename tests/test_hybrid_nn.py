import unittest

try:
    import torch
except ImportError:
    raise unittest.SkipTest("optional PyTorch dependency is not installed")

from speechprefix.hybrid_nn import SpeechPlanConfig, SpeechPlanHybrid


class SpeechPlanHybridTests(unittest.TestCase):
    def test_joint_loss_trains_planner_and_cross_attention_decoder(self):
        torch.set_num_threads(1)
        config = SpeechPlanConfig(
            vocab_size=40,
            pad_id=0,
            bos_id=2,
            eos_id=3,
            context_length=8,
            target_length=8,
            question_length=4,
            evidence_length=4,
            width=32,
            layers=2,
            heads=4,
            codebook_size=8,
            levels=(1, 2, 4, 8),
            dropout=0.0,
        )
        model = SpeechPlanHybrid(config)
        prefix = torch.tensor([[0, 0, 0, 0, 2, 4, 5, 6]])
        target = torch.randint(4, 40, (1, 8))
        question = torch.tensor([[7, 8, 0, 0]])
        evidence = torch.tensor([[9, 10, 0, 0]])
        losses = model(prefix, target, question, evidence)
        self.assertTrue(torch.isfinite(losses["loss"]))
        losses["loss"].backward()
        cross_attention = model.decoder.transformer.h[0].crossattention.c_attn.weight.grad
        self.assertIsNotNone(cross_attention)
        self.assertGreater(float(cross_attention.abs().sum()), 0.0)
        self.assertIsNotNone(model.code_heads[0].weight.grad)
        self.assertIsNotNone(model.question_summary_projection.weight.grad)
        self.assertIsNotNone(model.evidence_summary_projection.weight.grad)

        model.eval()
        generated = model.generate(
            prefix, max_new_tokens=3, top_k=2, question_ids=question, evidence_ids=evidence
        )
        self.assertLessEqual(len(generated), 3)
        self.assertTrue(all(0 <= token < config.vocab_size for token in generated))

    def test_summary_uses_last_nonpadding_token_for_left_padded_prefixes(self):
        config = SpeechPlanConfig(
            vocab_size=10, pad_id=0, bos_id=2, eos_id=3, context_length=4,
            target_length=4, width=8, layers=1, heads=2, codebook_size=4,
            levels=(1, 2, 4), dropout=0.0,
        )
        model = SpeechPlanHybrid(config)
        hidden = torch.tensor([[[0.0], [1.0], [2.0], [3.0]]])
        mask = torch.tensor([[False, False, True, True]])
        self.assertEqual(float(model._summary(hidden, mask)[0, 0]), 3.0)


if __name__ == "__main__":
    unittest.main()
