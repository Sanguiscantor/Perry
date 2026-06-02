# Cross-Asset Report (Phase 2)

## Train 4 / Test 1

| train | test | balanced_acc | ci_low | pass |
| --- | --- | --- | --- | --- |
| BTC,ETH,SOL,BNB | XRP | 0.5768 | 0.5638 | True |

## Leave-One-Out

| train | test | balanced_acc | ci_low | pass |
| --- | --- | --- | --- | --- |
| ETH,SOL,BNB,XRP | BTC | 0.5452 | 0.5468 | False |
| BTC,SOL,BNB,XRP | ETH | 0.6020 | 0.5887 | True |
| BTC,ETH,BNB,XRP | SOL | 0.5786 | 0.5670 | True |
| BTC,ETH,SOL,XRP | BNB | 0.5957 | 0.5941 | True |
| BTC,ETH,SOL,BNB | XRP | 0.5768 | 0.5626 | True |


## Decision Point B

- **Generalized:** True
- **Focus BTC only:** False
