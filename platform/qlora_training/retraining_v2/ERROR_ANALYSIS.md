# First-Run Error Analysis

The first adapter made 7 errors in 120 synthetic cases. Four errors were rejected by its validation-selected confidence gate; three were accepted. The 110 accepted predictions were 107/110 correct (97.27%). This is separate from unconditional accuracy, 113/120 (94.17%).

| Observed request | Expected | Predicted | Accepted |
|---|---|---|---|
| Unexpected authentication approvals without a login attempt | security | it | no |
| Unauthorized mailbox forwarding | security | it | no |
| Attachment executes a script and changes filenames | security | data | no |
| Application-code regression breaks valid image processing | production | data | no |
| Fiction-writing request mentioning a failed export | other | data | yes |
| Warehouse freshness blocked by a source watermark | data | production | yes |
| Approved subnet excluded by vault firewall configuration | cloud | security | yes |

## Response

Add new training scenarios for intent and symptom distinctions, not copies of these seven sentences. Keep their complete original 120-case cohort as development data. Remove repetitive formatting examples, regularize the adapter, and focus loss on domain tokens. Compare the original and new adapter under identical scoring on a new held-out synthetic cohort.

The new test is still model-authored and unreviewed. It is not a substitute for independent human labels or real-world incident evaluation. Improvement on a changed test cannot be claimed by comparing its percentage directly to the old 94.17%; compare both adapters on the new test instead.
