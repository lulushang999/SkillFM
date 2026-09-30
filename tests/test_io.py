"""JSONL shard ordering, source identity, and training compatibility."""
import hashlib
from pathlib import Path
import tempfile
import unittest

from skillflow.io import load_config, read_jsonl, sha256, training_rows
from skillflow.train_codec import prepare_data


class ShardedInputTest(unittest.TestCase):
    def test_filename_order_and_exact_byte_hash(self):
        first = b'{"task_id":"first","text":"a"}\n'
        second = b'{"task_id":"second","text":"b"}\n'
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parts = root / 'parts'
            parts.mkdir()
            (parts / 'part-001.jsonl').write_bytes(second)
            (parts / 'part-000.jsonl').write_bytes(first)
            whole = root / 'whole.jsonl'
            whole.write_bytes(first + second)
            self.assertEqual(read_jsonl(parts), read_jsonl(whole))
            self.assertEqual([row['task_id'] for row in read_jsonl(parts)], ['first', 'second'])
            self.assertEqual(sha256(parts), sha256(whole))
            self.assertEqual(sha256(parts), hashlib.sha256(first + second).hexdigest())

    def test_duplicate_id_across_shards_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for i in range(2):
                (root / f'part-{i:03d}.jsonl').write_text('{"task_id":"same"}\n')
            with self.assertRaisesRegex(ValueError, 'unique'):
                read_jsonl(root)

    def test_empty_shard_directory_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            for function in (read_jsonl, sha256):
                with self.assertRaisesRegex(ValueError, 'no JSONL shards'):
                    function(tmp)

    def test_alfworld_source_identity_and_codec_split(self):
        cfg = load_config('configs/alfworld.yaml')
        source = Path(cfg['data']['skills'])
        self.assertTrue(source.is_dir())
        parts = sorted(source.glob('*.jsonl'))
        self.assertEqual([p.name for p in parts], ['part-000.jsonl', 'part-001.jsonl'])
        self.assertTrue(all(p.stat().st_size < 5_000_000 for p in parts))
        self.assertFalse(source.with_suffix('.jsonl').exists())
        self.assertEqual(sha256(source), '628468cdb64f89071a8648b1df2c42b12afe0609cfaa55d1ae9bebcc2800ab43')
        rows = training_rows(cfg)
        self.assertEqual(len(rows), 3553)
        self.assertEqual([r['skill_id'] for r in rows], [f'alfworld_{i:06d}' for i in range(1, 3554)])
        _, train, validation, _, summary = prepare_data(cfg, 3)
        self.assertEqual((summary['codec_rows'], len(train), len(validation)), (3081, 2776, 305))
        self.assertEqual(summary['steps_per_epoch'], 58)


if __name__ == '__main__':
    unittest.main()
