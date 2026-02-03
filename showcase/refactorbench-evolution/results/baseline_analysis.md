# RefactorBench Baseline Results

**Date:** 2026-02-02
**Model:** Claude Sonnet (via Claude Code CLI)
**Strategy:** baseline_strategy.json (vanilla, no special instructions)

## Summary

| Metric | Value |
|--------|-------|
| **Pass Rate** | **53/100 (53.0%)** |
| **vs SOTA (35%)** | **+18 pts / 1.51x** |
| **vs Human (87%)** | 34 pts gap remaining |
| **Runtime** | 9096s (~2.5 hrs) |
| **Avg Turns/Task** | 16.4 |

## Comparison to Published Results (ICLR 2025 Paper)

| Method | Pass Rate | vs Our Baseline |
|--------|-----------|-----------------|
| Human | 87% | +34 pts |
| **Our Baseline (Sonnet + Claude Code)** | **53%** | — |
| Claude 3.5 Sonnet (descriptive) | 35% | -18 pts |
| SWE-agent + GPT-4 (base) | 22% | -31 pts |
| GPT-4 (descriptive) | 19% | -34 pts |
| Claude 3.5 Sonnet (base) | 17% | -36 pts |

## Per-Repository Breakdown

| Repository | Pass | Total | Rate | Notes |
|------------|------|-------|------|-------|
| salt_refactor | 12 | 15 | **80.0%** | Best performer |
| celery_refactor | 8 | 12 | 66.7% | Strong |
| fastapi_refactor | 4 | 6 | 66.7% | Strong |
| flask_refactor | 4 | 6 | 66.7% | Strong |
| ansible_refactor | 7 | 11 | 63.6% | Good |
| requests_refactor | 6 | 10 | 60.0% | Good |
| django_refactor | 10 | 18 | 55.6% | Average |
| tornado_refactor | 5 | 9 | 55.6% | Average |
| **scrapy_refactor** | **0** | **13** | **0.0%** | **Complete failure** |

**Without scrapy:** 53/87 = 60.9% pass rate

## Running Average Progression

```
#    Repo/Task                                                    Result   Running Avg  Pass/Done
1    ansible_refactor/add-log-parameter-get-group-vars            PASS     100.0%       1/1
2    ansible_refactor/add-log-parameter-is-systemd-managed        PASS     100.0%       2/2
3    ansible_refactor/combine-namespace-compat                    FAIL      66.7%       2/3
4    ansible_refactor/data-to-inventory-data                      PASS      75.0%       3/4
5    ansible_refactor/move-quoting-splitter                       FAIL      60.0%       3/5
6    ansible_refactor/new-inventory-patterns                      PASS      66.7%       4/6
7    ansible_refactor/new-utils-class-connection                  FAIL      57.1%       4/7
8    ansible_refactor/new-utils-from-basic                        FAIL      50.0%       4/8
9    ansible_refactor/parse_key_value                             PASS      55.6%       5/9
10   ansible_refactor/rename-lenient-lowercase                    PASS      60.0%       6/10
11   ansible_refactor/sort-groups-to-group-sort                   PASS      63.6%       7/11
12   celery_refactor/add-log-parameter-get-digest-algorithm       PASS      66.7%       8/12
13   celery_refactor/add-log-parameter-node-format                PASS      69.2%       9/13
14   celery_refactor/annotation-utils                             FAIL      64.3%       9/14
15   celery_refactor/autoretry-to-retry                           PASS      66.7%      10/15
16   celery_refactor/combine-unpickle-task                        FAIL      62.5%      10/16
17   celery_refactor/dump-message-to-serialization                PASS      64.7%      11/17
18   celery_refactor/ensure_serialize                             PASS      66.7%      12/18
19   celery_refactor/evaluate-promises-to-serialization           PASS      68.4%      13/19
20   celery_refactor/expand-router-string-to-utils                FAIL      65.0%      13/20
21   celery_refactor/object-mro-lookup                            PASS      66.7%      14/21
22   celery_refactor/rename-host-format                           FAIL      63.6%      14/22
23   celery_refactor/truncate-text                                FAIL      60.9%      14/23
24   django_refactor/add-log-parameter-constant-time-compare      PASS      62.5%      15/24
25   django_refactor/add-log-parameter-get-resolver               FAIL      60.0%      15/25
26   django_refactor/add-log-parameter-resolve-error-handler      FAIL      57.7%      15/26
27   django_refactor/add-none-handling-duration-string            FAIL      55.6%      15/27
28   django_refactor/combine-utils-dates-dateformat               PASS      57.1%      16/28
29   django_refactor/combine-utils-hashable-itercompat            FAIL      55.2%      16/29
30   django_refactor/new-converter-to-python-class                PASS      56.7%      17/30
31   django_refactor/new-path-traversal-exception                 FAIL      54.8%      17/31
32   django_refactor/new-reference-context-field-class            PASS      56.2%      18/32
33   django_refactor/new-reference-context-graph-class            FAIL      54.5%      18/33
34   django_refactor/new-timezone-class                           FAIL      52.9%      18/34
35   django_refactor/new-utils-adapt-method-mode                  PASS      54.3%      19/35
36   django_refactor/new-utils-check-response                     PASS      55.6%      20/36
37   django_refactor/new-utils-path-from-module                   PASS      56.8%      21/37
38   django_refactor/remove-core-cache-utils                      PASS      57.9%      22/38
39   django_refactor/remove-db-models-constants                   FAIL      56.4%      22/39
40   django_refactor/rename-file-move-safe                        FAIL      55.0%      22/40
41   django_refactor/split-parse-apps-and-model-labels            FAIL      53.7%      22/41
42   fastapi_refactor/add-log-parameter-generate-option-id-for-path PASS    54.8%      23/42
43   fastapi_refactor/exception-handlers-to-handlers              PASS      55.8%      24/43
44   fastapi_refactor/get-auth-scheme-param                       PASS      56.8%      25/44
45   fastapi_refactor/openapi-get-utils                           FAIL      55.6%      25/45
46   fastapi_refactor/params-to-param                             FAIL      54.3%      25/46
47   fastapi_refactor/value-is-a-sequence                         PASS      55.3%      26/47
48   flask_refactor/add-log-parameter-get-debug-flag              PASS      56.2%      27/48
49   flask_refactor/add-log-parameter-get-flashed-messages        PASS      57.1%      28/49
50   flask_refactor/debughelpers-to-helpers.py                    FAIL      56.0%      28/50
51   flask_refactor/rename-send-from-directory                    FAIL      54.9%      28/51
52   flask_refactor/render-template-str                           PASS      55.8%      29/52
53   flask_refactor/stream-template-str                           PASS      56.6%      30/53
54   requests_refactor/add-log-parameter-get-encoding-from-headers PASS     57.4%      31/54
55   requests_refactor/add-log-parameter-resolve-proxies          PASS      58.2%      32/55
56   requests_refactor/add-log-parameter-select-proxy             PASS      58.9%      33/56
57   requests_refactor/combine-from-key-to-key                    FAIL      57.9%      33/57
58   requests_refactor/combine-internal-utils-utils               FAIL      56.9%      33/58
59   requests_refactor/move-hooks-sessions                        FAIL      55.9%      33/59
60   requests_refactor/new-cookie-utils-class                     FAIL      55.0%      33/60
61   requests_refactor/rename-lookup-dict-dict-lookup             PASS      55.7%      34/61
62   requests_refactor/rename-super-len-complex-len               PASS      56.5%      35/62
63   requests_refactor/split-warnings-exceptions                  PASS      57.1%      36/63
64   salt_refactor/add-log-parameter-delete-directory             PASS      57.8%      37/64
65   salt_refactor/add-log-parameter-get-capability-definitions   PASS      58.5%      38/65
66   salt_refactor/add-log-parameter-recursive-diff               FAIL      57.6%      38/66
67   salt_refactor/cant-create                                    PASS      58.2%      39/67
68   salt_refactor/channel-to-transport                           PASS      58.8%      40/68
69   salt_refactor/ex-pillar-fail                                 PASS      59.4%      41/69
70   salt_refactor/ex-state-fail                                  PASS      60.0%      42/70
71   salt_refactor/exactly-n-boto-mod                             PASS      60.6%      43/71
72   salt_refactor/get-unavail                                    PASS      61.1%      44/72
73   salt_refactor/iam-to-aws                                     PASS      61.6%      45/73
74   salt_refactor/mksls-to-specific                              PASS      62.2%      46/74
75   salt_refactor/namecheap-xmlutil                              PASS      62.7%      47/75
76   salt_refactor/paged-call-boto-mod                            FAIL      61.8%      47/76
77   salt_refactor/pem-fingerprint                                FAIL      61.0%      47/77
78   salt_refactor/perm-denied                                    PASS      61.5%      48/78
79   scrapy_refactor/add-log-parameter-disconnect-all             FAIL      60.8%      48/79
80   scrapy_refactor/add-log-parameter-job-dir                    FAIL      60.0%      48/80
81   scrapy_refactor/add-log-parameter-xmliter                    FAIL      59.3%      48/81
82   scrapy_refactor/genspider-functions-to-utils-url             FAIL      58.5%      48/82
83   scrapy_refactor/new-downloadermiddlewares-utils              FAIL      57.8%      48/83
84   scrapy_refactor/new-spider-utils-in-spiders                  FAIL      57.1%      48/84
85   scrapy_refactor/new-verify-reactor-class                     FAIL      56.5%      48/85
86   scrapy_refactor/not-supported-exception-to-unsupported       FAIL      55.8%      48/86
87   scrapy_refactor/parameterize-gunzip                          FAIL      55.2%      48/87
88   scrapy_refactor/rename-description-commands                  FAIL      54.5%      48/88
89   scrapy_refactor/rename-engine-status                         FAIL      53.9%      48/89
90   scrapy_refactor/rename-processtest-testproc                  FAIL      53.3%      48/90
91   scrapy_refactor/sitemap-url-to-url                           FAIL      52.7%      48/91
92   tornado_refactor/global-objects                              FAIL      52.2%      48/92
93   tornado_refactor/log-utils                                   PASS      52.7%      49/93
94   tornado_refactor/option-parser-with-pretty-print             PASS      53.2%      50/94
95   tornado_refactor/options-utils                               FAIL      52.6%      50/95
96   tornado_refactor/remove-locale-data                          PASS      53.1%      51/96
97   tornado_refactor/rename-http1connection                      PASS      53.6%      52/97
98   tornado_refactor/rename-to-camel-case                        PASS      54.1%      53/98
99   tornado_refactor/resolvers-as-separate                       FAIL      53.5%      53/99
100  tornado_refactor/tcpclient-connect-params                    FAIL      53.0%      53/100
```

## Task-Level Results

### Passed Tasks (53)

| # | Repository | Task | Turns |
|---|------------|------|-------|
| 1 | ansible_refactor | add-log-parameter-get-group-vars | 16 |
| 2 | ansible_refactor | add-log-parameter-is-systemd-managed | 16 |
| 4 | ansible_refactor | data-to-inventory-data | 26 |
| 6 | ansible_refactor | new-inventory-patterns | 16 |
| 9 | ansible_refactor | parse_key_value | 16 |
| 10 | ansible_refactor | rename-lenient-lowercase | 27 |
| 11 | ansible_refactor | sort-groups-to-group-sort | 16 |
| 12 | celery_refactor | add-log-parameter-get-digest-algorithm | 16 |
| 13 | celery_refactor | add-log-parameter-node-format | 16 |
| 15 | celery_refactor | autoretry-to-retry | 16 |
| 17 | celery_refactor | dump-message-to-serialization | 16 |
| 18 | celery_refactor | ensure_serialize | 21 |
| 19 | celery_refactor | evaluate-promises-to-serialization | 21 |
| 21 | celery_refactor | object-mro-lookup | 12 |
| 24 | django_refactor | add-log-parameter-constant-time-compare | 16 |
| 28 | django_refactor | combine-utils-dates-dateformat | 16 |
| 30 | django_refactor | new-converter-to-python-class | 22 |
| 32 | django_refactor | new-reference-context-field-class | 16 |
| 35 | django_refactor | new-utils-adapt-method-mode | 16 |
| 36 | django_refactor | new-utils-check-response | 16 |
| 37 | django_refactor | new-utils-path-from-module | 16 |
| 38 | django_refactor | remove-core-cache-utils | 16 |
| 42 | fastapi_refactor | add-log-parameter-generate-option-id-for-path | 14 |
| 43 | fastapi_refactor | exception-handlers-to-handlers | 16 |
| 44 | fastapi_refactor | get-auth-scheme-param | 16 |
| 47 | fastapi_refactor | value-is-a-sequence | 19 |
| 48 | flask_refactor | add-log-parameter-get-debug-flag | 16 |
| 49 | flask_refactor | add-log-parameter-get-flashed-messages | 16 |
| 52 | flask_refactor | render-template-str | 16 |
| 53 | flask_refactor | stream-template-str | 13 |
| 54 | requests_refactor | add-log-parameter-get-encoding-from-headers | 19 |
| 55 | requests_refactor | add-log-parameter-resolve-proxies | 16 |
| 56 | requests_refactor | add-log-parameter-select-proxy | 16 |
| 61 | requests_refactor | rename-lookup-dict-dict-lookup | 16 |
| 62 | requests_refactor | rename-super-len-complex-len | 16 |
| 63 | requests_refactor | split-warnings-exceptions | 16 |
| 64 | salt_refactor | add-log-parameter-delete-directory | 21 |
| 65 | salt_refactor | add-log-parameter-get-capability-definitions | 17 |
| 67 | salt_refactor | cant-create | 19 |
| 68 | salt_refactor | channel-to-transport | 16 |
| 69 | salt_refactor | ex-pillar-fail | 20 |
| 70 | salt_refactor | ex-state-fail | 18 |
| 71 | salt_refactor | exactly-n-boto-mod | 19 |
| 72 | salt_refactor | get-unavail | 21 |
| 73 | salt_refactor | iam-to-aws | 15 |
| 74 | salt_refactor | mksls-to-specific | 16 |
| 75 | salt_refactor | namecheap-xmlutil | 16 |
| 78 | salt_refactor | perm-denied | 16 |
| 93 | tornado_refactor | log-utils | 20 |
| 94 | tornado_refactor | option-parser-with-pretty-print | 16 |
| 96 | tornado_refactor | remove-locale-data | 13 |
| 97 | tornado_refactor | rename-http1connection | 16 |
| 98 | tornado_refactor | rename-to-camel-case | 16 |

### Failed Tasks (47)

| # | Repository | Task | Turns |
|---|------------|------|-------|
| 3 | ansible_refactor | combine-namespace-compat | 16 |
| 5 | ansible_refactor | move-quoting-splitter | 16 |
| 7 | ansible_refactor | new-utils-class-connection | 16 |
| 8 | ansible_refactor | new-utils-from-basic | 16 |
| 14 | celery_refactor | annotation-utils | 16 |
| 16 | celery_refactor | combine-unpickle-task | 16 |
| 20 | celery_refactor | expand-router-string-to-utils | 16 |
| 22 | celery_refactor | rename-host-format | 16 |
| 23 | celery_refactor | truncate-text | 16 |
| 25 | django_refactor | add-log-parameter-get-resolver | 16 |
| 26 | django_refactor | add-log-parameter-resolve-error-handler | 16 |
| 27 | django_refactor | add-none-handling-duration-string | 16 |
| 29 | django_refactor | combine-utils-hashable-itercompat | 16 |
| 31 | django_refactor | new-path-traversal-exception | 16 |
| 33 | django_refactor | new-reference-context-graph-class | 16 |
| 34 | django_refactor | new-timezone-class | 16 |
| 39 | django_refactor | remove-db-models-constants | 16 |
| 40 | django_refactor | rename-file-move-safe | 16 |
| 41 | django_refactor | split-parse-apps-and-model-labels | 16 |
| 45 | fastapi_refactor | openapi-get-utils | 16 |
| 46 | fastapi_refactor | params-to-param | 16 |
| 50 | flask_refactor | debughelpers-to-helpers.py | 16 |
| 51 | flask_refactor | rename-send-from-directory | 16 |
| 57 | requests_refactor | combine-from-key-to-key | 16 |
| 58 | requests_refactor | combine-internal-utils-utils | 16 |
| 59 | requests_refactor | move-hooks-sessions | 16 |
| 60 | requests_refactor | new-cookie-utils-class | 16 |
| 66 | salt_refactor | add-log-parameter-recursive-diff | 16 |
| 76 | salt_refactor | paged-call-boto-mod | 14 |
| 77 | salt_refactor | pem-fingerprint | 16 |
| 79 | scrapy_refactor | add-log-parameter-disconnect-all | 16 |
| 80 | scrapy_refactor | add-log-parameter-job-dir | 16 |
| 81 | scrapy_refactor | add-log-parameter-xmliter | 16 |
| 82 | scrapy_refactor | genspider-functions-to-utils-url | 16 |
| 83 | scrapy_refactor | new-downloadermiddlewares-utils | 16 |
| 84 | scrapy_refactor | new-spider-utils-in-spiders | 16 |
| 85 | scrapy_refactor | new-verify-reactor-class | 16 |
| 86 | scrapy_refactor | not-supported-exception-to-unsupported | 16 |
| 87 | scrapy_refactor | parameterize-gunzip | 16 |
| 88 | scrapy_refactor | rename-description-commands | 16 |
| 89 | scrapy_refactor | rename-engine-status | 16 |
| 90 | scrapy_refactor | rename-processtest-testproc | 16 |
| 91 | scrapy_refactor | sitemap-url-to-url | 16 |
| 92 | tornado_refactor | global-objects | 16 |
| 95 | tornado_refactor | options-utils | 16 |
| 99 | tornado_refactor | resolvers-as-separate | 16 |
| 100 | tornado_refactor | tcpclient-connect-params | 16 |

## Failure Analysis

### By Task Type Pattern

| Pattern | Failed | Total | Fail Rate |
|---------|--------|-------|-----------|
| new-* (create new class/module) | 9 | 15 | 60% |
| combine-* (merge modules) | 5 | 6 | 83% |
| rename-* | 6 | 14 | 43% |
| add-log-parameter-* | 5 | 16 | 31% |
| move-* | 2 | 2 | 100% |
| split-* | 2 | 2 | 100% |
| remove-* | 1 | 2 | 50% |

### Key Observations

1. **Scrapy complete failure (0/13)**: Likely due to test structure or codebase complexity
2. **Most failures hit max turns (16)**: Agent ran out of attempts rather than early failures
3. **"combine" and "move" tasks hardest**: Require understanding multiple file dependencies
4. **"add-log-parameter" tasks easiest**: Simple signature changes

## Evolution Opportunities

1. **Scrapy-specific handling**: 13 easy wins if we can crack the pattern
2. **Better file discovery for "combine" tasks**: Find all dependent files before editing
3. **State tracking for multi-file edits**: Track which files modified, which remain
4. **Incremental validation**: Run tests after each edit to catch issues early
