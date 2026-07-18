from __future__ import annotations

import importlib.util
import sys
import unittest
from unittest import mock
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_builder():
    path = PROJECT_ROOT / "scripts/revision/36_build_marked_manuscript.py"
    spec = importlib.util.spec_from_file_location("marked_manuscript_builder_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class MarkedManuscriptBuilderTests(unittest.TestCase):
    def test_default_original_is_previous_manuscript_not_ieee_template(self):
        builder = load_builder()
        with mock.patch.object(sys, "argv", ["test"]):
            original = builder.parse_args().original

        self.assertEqual(original.name, "main_pre_final_evidence_20260707.tex")
        self.assertNotEqual(original.name, "BACKUP.tex")

    def test_noalign_commands_are_not_prefixed_by_latexdiff_float_markers(self):
        builder = load_builder()
        source = """\\\\
\\DIFaddbeginFL \\midrule
\\DIFaddFL{row} \\\\
\\DIFaddbeginFL \\bottomrule
"""

        result = builder.sanitize_revised_view_diff(source)

        self.assertNotIn(r"\DIFaddbeginFL \midrule", result)
        self.assertNotIn(r"\DIFaddbeginFL \bottomrule", result)
        self.assertIn(r"\midrule", result)
        self.assertIn(r"\bottomrule", result)
        self.assertIn(r"\DIFaddFL{row}", result)

    def test_flattened_bibliography_is_restored_from_revised_source(self):
        builder = load_builder()
        marked = r"""body
\DIFaddbegin \begin{thebibliography}{10}
\bibitem{source}\DIFadd{A source.}\newblock \DIFadd{Accessed.}
\end{thebibliography}
\DIFaddend
\end{document}
"""
        revised = r"""body
\bibliography{main}
\end{document}
"""

        result = builder.restore_revised_bibliography_command(marked, revised)

        self.assertIn(r"\bibliography{main}", result)
        self.assertNotIn(r"\begin{thebibliography}", result)
        self.assertNotIn(r"\DIFadd{Accessed.}", result)
        self.assertIn(r"\end{document}", result)


if __name__ == "__main__":
    unittest.main()
