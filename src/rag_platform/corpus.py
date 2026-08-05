"""Synthetic enterprise corpus with labelled query→document relevance.

Built rather than downloaded because public IR benchmarks (MS MARCO, BEIR) are
web text, and the claim under test is specifically about *enterprise* documents:
dense with part numbers, error codes, policy identifiers, and internal acronyms.
A benchmark that does not contain those cannot demonstrate why lexical retrieval
still matters.

The corpus is deliberately adversarial toward pure dense retrieval in a way real
corpora are:

- **Identifier queries** — `RMA-4471`, `ERR-5012`. Embeddings place all
  identifiers in roughly the same region; only exact matching separates them.
- **Near-duplicate versions** — the same policy at v1 and v2, one superseded.
  Retrieval must prefer the current one.
- **Acronym collisions** — `PTO` as paid time off and as a hardware term, so a
  purely semantic match can land in the wrong department.
- **Paraphrase queries** — no lexical overlap at all, where dense should win and
  BM25 should fail. Included so the comparison is honest rather than rigged.

Every document is authored fiction. No real company, person, or policy.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .models import Document

__all__ = [
    "EvalQuery",
    "build_corpus",
    "build_distractors",
    "build_full_corpus",
    "build_queries",
]


@dataclass
class EvalQuery:
    """A query with its ground-truth relevant document IDs."""

    id: str
    text: str
    relevant_doc_ids: set[str]
    kind: str  # identifier | paraphrase | acronym | versioned | mixed
    roles: set[str] | None = None
    note: str = ""
    metadata: dict[str, str] = field(default_factory=dict)


_DOCS: list[tuple[str, str, str, str, tuple[str, ...], int, bool]] = [
    # (id, title, doc_type, text, allowed_roles, version, superseded)
    (
        "pol-leave-v1",
        "Annual Leave Policy (v1, superseded)",
        "policy",
        "Employees accrue 20 days of annual leave per calendar year. Unused leave "
        "may be carried over to a maximum of 5 days. Carry-over must be used by "
        "31 March of the following year or it is forfeited. Requests are submitted "
        "through the HR portal at least 10 working days in advance.",
        ("*",),
        1,
        True,
    ),
    (
        "pol-leave-v2",
        "Annual Leave Policy (v2, current)",
        "policy",
        "Employees accrue 25 days of annual leave per calendar year. Unused leave "
        "may be carried over to a maximum of 10 days. Carry-over must be used by "
        "30 June of the following year or it is forfeited. Requests are submitted "
        "through the HR portal at least 5 working days in advance. This policy "
        "supersedes the previous annual leave policy.",
        ("*",),
        2,
        False,
    ),
    (
        "rma-4471",
        "RMA-4471 — Thermal throttling on X200 units",
        "runbook",
        "RMA-4471 covers X200 chassis units manufactured before batch 88C that "
        "exhibit thermal throttling under sustained load. Affected units report "
        "core temperatures above 92C within 40 minutes. Replacement heatsink kit "
        "part number HSK-9902 resolves the issue. Units under warranty are "
        "replaced at no cost; out-of-warranty repairs are quoted separately.",
        ("*",),
        1,
        False,
    ),
    (
        "rma-4482",
        "RMA-4482 — Power supply failure on X200 units",
        "runbook",
        "RMA-4482 covers X200 chassis units that fail to power on after a firmware "
        "update. Root cause is a corrupted bootloader on the secondary PSU "
        "controller. Recovery requires reflashing with tool FWFLASH-3. Part number "
        "PSU-4410 is the replacement unit if reflashing fails.",
        ("*",),
        1,
        False,
    ),
    (
        "err-5012",
        "ERR-5012 — Payment authorization declined",
        "runbook",
        "Error ERR-5012 is returned when the acquiring bank declines an "
        "authorization without a specific reason code. Common causes are "
        "insufficient funds, a card-level velocity limit, or an issuer fraud rule. "
        "Retrying the same transaction within 60 seconds will produce the same "
        "result. Advise the customer to contact their issuing bank.",
        ("*",),
        1,
        False,
    ),
    (
        "err-5013",
        "ERR-5013 — Payment gateway timeout",
        "runbook",
        "Error ERR-5013 indicates the payment gateway did not respond within the "
        "8 second timeout window. The transaction state is indeterminate and must "
        "be reconciled before retrying, or a duplicate charge may result. Use the "
        "reconciliation endpoint to confirm final state before any retry.",
        ("*",),
        1,
        False,
    ),
    (
        "hr-pto-carryover",
        "PTO carry-over guidance",
        "guidance",
        "PTO refers to paid time off. Carry-over of PTO is governed by the current "
        "annual leave policy. Managers may approve exceptional carry-over beyond "
        "the standard limit only with written HR approval, and only where the "
        "employee was prevented from taking leave by business need.",
        ("*",),
        1,
        False,
    ),
    (
        "hw-pto-connector",
        "PTO connector specification",
        "spec",
        "The PTO connector on the X200 chassis is a power take-off interface "
        "supplying 12V at up to 4A to auxiliary modules. PTO here denotes power "
        "take-off and is unrelated to human resources terminology. Maximum "
        "sustained draw is 48W; exceeding it triggers a protective cutout.",
        ("*",),
        1,
        False,
    ),
    (
        "sec-incident",
        "Security incident response — restricted",
        "policy",
        "On confirmed compromise of production credentials, the on-call security "
        "engineer must rotate all affected secrets within 30 minutes and open a "
        "P1 incident. Customer notification is decided by the incident commander "
        "in consultation with legal counsel. Do not discuss active incidents "
        "outside the incident channel.",
        ("security", "admin"),
        1,
        False,
    ),
    (
        "fin-expense",
        "Expense reimbursement — restricted to finance",
        "policy",
        "Expenses above 500 require a receipt and line-manager approval before "
        "submission. Reimbursement is processed in the payroll run following "
        "approval. Travel booked outside the approved provider is reimbursed at "
        "the lower of actual cost or the reference fare.",
        ("finance", "admin"),
        1,
        False,
    ),
    (
        "onboarding",
        "New joiner onboarding checklist",
        "guidance",
        "New joiners receive laptop provisioning on day one, access to the HR "
        "portal within 24 hours, and a buddy assignment in the first week. "
        "Mandatory security awareness training must be completed within 30 days "
        "of the start date.",
        ("*",),
        1,
        False,
    ),
    (
        "remote-work",
        "Remote working arrangements",
        "policy",
        "Employees may work remotely up to three days per week by default. Fully "
        "remote arrangements require director approval and a documented business "
        "case. Equipment for home working is provided against the standard "
        "allowance; monitors and chairs are covered, desks are not.",
        ("*",),
        1,
        False,
    ),
]


def build_corpus() -> list[Document]:
    """The full synthetic corpus."""
    return [
        Document(
            id=doc_id,
            title=title,
            doc_type=doc_type,
            text=text,
            source=f"synthetic://{doc_id}",
            allowed_roles=roles,
            version=version,
            superseded=superseded,
        )
        for doc_id, title, doc_type, text, roles, version, superseded in _DOCS
    ]


def build_queries() -> list[EvalQuery]:
    """Labelled queries spanning the retrieval failure modes worth measuring."""
    return [
        # --- Identifier queries: BM25 should win decisively --------------------
        EvalQuery(
            "q01", "RMA-4471", {"rma-4471"}, "identifier",
            note="Bare identifier. Embeddings cluster all RMA codes together.",
        ),
        EvalQuery(
            "q02", "what does ERR-5012 mean", {"err-5012"}, "identifier",
            note="Identifier plus natural language.",
        ),
        EvalQuery(
            "q03", "HSK-9902 part", {"rma-4471"}, "identifier",
            note="Part number appears only in the body, not the title.",
        ),
        EvalQuery(
            "q04", "FWFLASH-3 recovery tool", {"rma-4482"}, "identifier",
        ),
        # --- Paraphrase queries: dense should win, BM25 should struggle --------
        EvalQuery(
            "q05", "how many holiday days do I get each year", {"pol-leave-v2"}, "paraphrase",
            note="No lexical overlap with 'annual leave accrual'.",
        ),
        EvalQuery(
            "q06", "can I work from home most of the week", {"remote-work"}, "paraphrase",
        ),
        EvalQuery(
            "q07", "what happens on my first day at the company", {"onboarding"}, "paraphrase",
        ),
        EvalQuery(
            "q08", "the card was refused by the bank", {"err-5012"}, "paraphrase",
            note="Describes the symptom without naming the error code.",
        ),
        # --- Acronym collision: semantic match alone lands in the wrong domain -
        EvalQuery(
            "q09", "PTO carry over limit", {"hr-pto-carryover", "pol-leave-v2"}, "acronym",
            note="PTO as paid time off, not power take-off.",
        ),
        EvalQuery(
            "q10", "PTO connector power rating", {"hw-pto-connector"}, "acronym",
            note="Same acronym, hardware sense.",
        ),
        # --- Versioning: the superseded revision must not be returned ----------
        EvalQuery(
            "q11", "how much annual leave can I carry over", {"pol-leave-v2"}, "versioned",
            note="v1 says 5 days, v2 says 10. Returning v1 is a correctness bug.",
        ),
        EvalQuery(
            "q12", "how far in advance must I request leave", {"pol-leave-v2"}, "versioned",
        ),
        # --- Mixed: both legs contribute --------------------------------------
        EvalQuery(
            "q13", "X200 overheating under load", {"rma-4471"}, "mixed",
        ),
        EvalQuery(
            "q14", "duplicate charge risk after gateway timeout", {"err-5013"}, "mixed",
        ),
        EvalQuery(
            "q15", "X200 will not power on after firmware update", {"rma-4482"}, "mixed",
        ),
        # --- Access control: correct answer exists but must be withheld -------
        EvalQuery(
            "q16", "what do I do if production credentials leak", {"sec-incident"},
            "mixed", roles={"security"},
            note="Visible to security role.",
        ),
        EvalQuery(
            "q17", "what do I do if production credentials leak", set(),
            "mixed", roles={"engineering"},
            note="Same query, unprivileged role. The correct result is NOTHING — "
                 "a system that answers here has leaked restricted content.",
        ),
        EvalQuery(
            "q18", "expense approval threshold", set(), "mixed", roles={"engineering"},
            note="Finance-restricted; must return nothing for this role.",
        ),
        EvalQuery(
            "q19", "expense approval threshold", {"fin-expense"}, "mixed", roles={"finance"},
        ),
        EvalQuery(
            "q20", "security awareness training deadline", {"onboarding"}, "mixed",
        ),
    ]


# --- Distractor layer -------------------------------------------------------
#
# The twelve documents above are the labelled gold set. On their own they are
# not a benchmark: with a corpus that small, retrieving k=5 fetches 42% of
# everything and recall@5 is 1.000 for every strategy — the measurement cannot
# discriminate, which is a property of the corpus rather than the retrievers.
#
# These generated documents are the fix. They are deliberately *near misses*:
# hundreds of runbooks whose identifiers differ by a digit, policies on adjacent
# topics, specs sharing vocabulary. That is what makes identifier queries hard
# for dense retrieval — an embedding places RMA-4471 and RMA-4472 in nearly the
# same place, and only exact matching separates them.

_SYMPTOMS = [
    "intermittent packet loss on the management interface",
    "fan controller reporting an implausible RPM value",
    "slow disk writes after a firmware rollback",
    "memory errors logged during extended self-test",
    "display output dropping to a lower refresh rate",
    "USB peripherals disconnecting under sustained load",
    "boot delay of more than ninety seconds",
    "network link negotiating at a reduced speed",
]
_REMEDIES = [
    "Replace the affected module and re-run diagnostics.",
    "Apply the latest firmware bundle and reboot twice.",
    "Reseat the connector and verify the retention clip is engaged.",
    "Swap the cable for a certified replacement and re-test.",
    "Clear the event log, run a full self-test, and capture the report.",
]
_POLICY_TOPICS = [
    ("Training and development budget", "Employees may claim up to an annual budget for "
     "external training with line-manager approval."),
    ("Business travel booking", "Travel must be booked through the approved provider at "
     "least fourteen days before departure where practical."),
    ("Equipment loss and damage", "Loss or damage to company equipment must be reported "
     "to IT within one working day of discovery."),
    ("Probation review", "Probation reviews are held at the three month mark with a "
     "written outcome recorded in the HR portal."),
    ("Internal transfer", "Employees may apply for internal transfer after twelve months "
     "in role, subject to manager consultation."),
    ("Contractor engagement", "Contractor engagements above the standard threshold "
     "require procurement review before a purchase order is raised."),
    ("Data classification", "Documents are classified as public, internal, confidential, "
     "or restricted, and handled according to the matrix."),
    ("Meeting room booking", "Rooms may be booked up to four weeks ahead; recurring "
     "bookings above two hours require facilities approval."),
]


def build_distractors(n_runbooks: int = 120, n_policies: int = 64) -> list[Document]:
    """Generated near-miss documents that make the benchmark discriminative.

    Identifiers are drawn to sit *close to but not on* the gold identifiers, so
    a retriever that merely finds "something that looks like an RMA code" scores
    zero. Deterministic, so the benchmark is reproducible.
    """
    import random

    rng = random.Random(20260805)
    docs: list[Document] = []

    for i in range(n_runbooks):
        # Avoid colliding with the gold identifiers (4471, 4482, 5012, 5013).
        code = 4400 + i if 4400 + i not in (4471, 4482) else 4400 + i + 200
        family = "RMA" if i % 2 == 0 else "ERR"
        if family == "ERR":
            code = 5000 + i if 5000 + i not in (5012, 5013) else 5000 + i + 200
        symptom = _SYMPTOMS[i % len(_SYMPTOMS)]
        remedy = _REMEDIES[i % len(_REMEDIES)]
        part = f"{'HSK' if i % 3 == 0 else 'PSU' if i % 3 == 1 else 'CBL'}-{7000 + i}"
        docs.append(
            Document(
                id=f"gen-{family.lower()}-{code}",
                title=f"{family}-{code} — {symptom[:38]}",
                doc_type="runbook",
                text=(
                    f"{family}-{code} covers units exhibiting {symptom}. Affected "
                    f"hardware is identified by the service tag range recorded in the "
                    f"asset register. {remedy} Replacement part number {part} is stocked "
                    f"in the regional depot. Escalate to tier two if the symptom persists "
                    f"after two remediation attempts."
                ),
                source=f"synthetic://gen-{family.lower()}-{code}",
                allowed_roles=("*",),
            )
        )

    for i in range(n_policies):
        title, body = _POLICY_TOPICS[i % len(_POLICY_TOPICS)]
        year = 2023 + (i % 3)
        docs.append(
            Document(
                id=f"gen-pol-{i:03d}",
                title=f"{title} ({year} revision)",
                doc_type="policy",
                text=(
                    f"{body} This revision took effect in {year} and applies to all "
                    f"permanent employees. Exceptions require written approval from the "
                    f"relevant department head. Questions should be directed to the "
                    f"people team through the HR portal."
                ),
                source=f"synthetic://gen-pol-{i:03d}",
                allowed_roles=("*",),
            )
        )

    rng.shuffle(docs)
    return docs


def build_full_corpus(with_distractors: bool = True) -> list[Document]:
    """Gold documents plus the distractor layer — the benchmark corpus."""
    docs = build_corpus()
    if with_distractors:
        docs = docs + build_distractors()
    return docs
