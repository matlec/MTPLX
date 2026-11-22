"""Inkling 2 image input on the inkling2_loop trunk (tiny random model):

* the vision tower builds from ``vision_config``, takes ``model.visual.*``
  through ``sanitize``, and stays BF16 and unquantized,
* ``MTPLX_INKLING2_VISION=0`` drops the tower and its weights,
* image rows spliced at the image tokens through the runtime's AR forward
  equal the model's own forward on the same input rows,
* the server keeps image parts for the template and expands each image's
  placeholder to its rows.
"""

import base64
import io
import os
import unittest
from types import SimpleNamespace
from unittest import mock

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from mtplx.models import inkling2_loop as ink
from mtplx.models.inkling2_vision import Inkling2ImageProcessor, encode_images
from tests.test_inkling2_loop import CONFIG

VISION_CONFIG = {
    "hidden_size": 32,
    "num_hidden_layers": 2,
    "num_attention_heads": 2,
    "native_patch_size": 4,
    "adapter_merge_size": 2,
    "posemb_grid_size": 4,
    "projection_hidden_dim": 48,
}
IMAGE_TOKEN = 100


def _vision_model(seed=0):
    mx.random.seed(seed)
    config = dict(CONFIG, vision_config=VISION_CONFIG, image_token_id=IMAGE_TOKEN)
    model = ink.Model(ink.ModelArgs.from_dict(config))
    params = [
        (k, mx.random.normal(v.shape) * 0.05) for k, v in tree_flatten(model.parameters())
    ]
    model.load_weights(params)
    mx.eval(model.parameters())
    return model


def _png(width=24, height=16) -> bytes:
    from PIL import Image

    image = Image.new("RGB", (width, height), "white")
    image.paste((255, 0, 0), (0, 0, width // 2, height))
    buf = io.BytesIO()
    image.save(buf, "PNG")
    return buf.getvalue()


class VisionLoadTest(unittest.TestCase):
    def test_sanitize_maps_visual_weights_and_skips_their_quantization(self):
        model = _vision_model()
        visual = {
            "model.visual." + k[len("vision_tower.") :]: v
            for k, v in tree_flatten(model.parameters())
            if k.startswith("vision_tower.")
        }
        out = model.sanitize(dict(visual))
        self.assertEqual(
            sorted(out), sorted("vision_tower." + k[len("model.visual.") :] for k in visual)
        )
        attn = model.vision_tower.backbone.layers[0].self_attn.q_proj
        self.assertIsInstance(attn, nn.Linear)
        self.assertNotIsInstance(attn, nn.QuantizedLinear)
        self.assertFalse(model.cast_predicate("vision_tower.backbone.ln_pre.weight"))
        self.assertTrue(model.cast_predicate("model.language_model.norm.weight"))

    def test_vision_off_drops_the_tower(self):
        with mock.patch.dict(os.environ, {"MTPLX_INKLING2_VISION": "0"}):
            config = dict(CONFIG, vision_config=VISION_CONFIG)
            model = ink.Model(ink.ModelArgs.from_dict(config))
        self.assertIsNone(getattr(model, "vision_tower", None))
        self.assertEqual(model.sanitize({"model.visual.proj_in.weight": mx.zeros((2, 2))}), {})


class VisionSpliceTest(unittest.TestCase):
    def test_runtime_forward_takes_spliced_rows(self):
        from mtplx.runtime import MTPLXRuntime
        from mtplx.vision.splice import VisionSplice, spliced_chunk_embeddings

        model = _vision_model()
        processor = Inkling2ImageProcessor(size={"height": 8, "width": 8})
        (rows,) = encode_images(model, processor, [_png()])
        self.assertEqual(rows.shape, (2 * 3, CONFIG["text_config"]["hidden_size"]))

        ids = [1, 2] + [IMAGE_TOKEN] * rows.shape[0] + [3, 4]
        runtime = MTPLXRuntime.__new__(MTPLXRuntime)
        runtime.model = model
        runtime.mtp_enabled = False
        runtime.diagnostic_counters = {}
        runtime._forward_ar_supports_emit_logits = None
        runtime._forward_ar_supports_logits_keep = None
        splice = VisionSplice(image_pad_token_id=IMAGE_TOKEN, embeddings=rows)
        chunk = mx.array([ids])
        embeddings = spliced_chunk_embeddings(runtime.embed_tokens, chunk, splice)
        self.assertEqual(splice.remaining(), 0)

        got = runtime.forward_ar(chunk, cache=model.make_cache(), input_embeddings=embeddings)
        want = model(chunk, cache=model.make_cache(), input_embeddings=embeddings)
        plain = model(chunk, cache=model.make_cache())
        self.assertTrue(mx.allclose(got, want).item())
        # the image rows changed the input (not the pads' token embeddings)
        self.assertFalse(mx.allclose(got, plain).item())
        # text rows are the normalized embeddings the trunk embeds itself
        self.assertTrue(
            mx.allclose(embeddings[0, :2], model.model.language_model.embed(chunk[:, :2])[0]).item()
        )


class ServerPartsTest(unittest.TestCase):
    def test_image_parts_reach_the_template(self):
        from mtplx.server import openai as server

        url = "data:image/png;base64," + base64.b64encode(_png()).decode()
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "what is this?"},
                    {"type": "image_url", "image_url": {"url": url}},
                ],
            }
        ]
        kept, images = server._vision_extract_and_flatten(messages, keep_parts=True)
        self.assertEqual(len(images), 1)
        self.assertEqual(kept[0]["content"][1], {"type": "image"})
        message = server.ChatMessage(role="user", content=kept[0]["content"])
        rendered = server._message_to_template_dict(
            message, strip_assistant_reasoning_history=False
        )
        self.assertEqual(
            rendered["content"], [{"type": "text", "text": "what is this?"}, {"type": "image"}]
        )

    def test_model_vision_spec(self):
        from mtplx.server import openai as server

        model = _vision_model()
        state = SimpleNamespace(
            runtime=SimpleNamespace(model=model), args=SimpleNamespace(model="/nonexistent")
        )
        spec = server._server_vision_spec(state)
        self.assertEqual(spec.image_token_id, IMAGE_TOKEN)
        self.assertTrue(server._vision_keeps_parts(state))
        spec.image_processor = Inkling2ImageProcessor(size={"height": 8, "width": 8})
        ids, splice = server._materialize_vision_splice(state, [_png()], [5, IMAGE_TOKEN, 6])
        self.assertEqual(ids, [5] + [IMAGE_TOKEN] * 6 + [6])
        self.assertEqual(splice.pad_counts, (6,))
        self.assertEqual(len(splice.image_digests), 1)


if __name__ == "__main__":
    unittest.main()
