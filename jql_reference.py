"""
JQL (Jira Query Language) reference used both as developer documentation
and injected verbatim into LLM prompts to constrain query generation.

Covers Jira Server v9 / REST API v2.
"""

JQL_REFERENCE = """
## JQL Reference (Jira Server v9)

### Basic Operators
| Operator       | Example                                              |
|----------------|------------------------------------------------------|
| =              | status = "In Progress"                               |
| != / not in    | issuetype != Sub-task                                |
| in (...)       | priority in (Major, Critical, Blocker)               |
| not in (...)   | status not in (Done, Closed, Resolved)               |
| is EMPTY       | assignee is EMPTY                                    |
| is not EMPTY   | resolutiondate is not EMPTY                          |
| >= / <=        | created >= "2024-01-01" AND created <= "2024-03-31"  |
| ~ (contains)   | summary ~ "login error"                              |
| !~ (not cont.) | summary !~ "test"                                    |

### Change History Operators
These require needs_changelog: true in the plan.

| Operator / Predicate        | Example                                                              |
|-----------------------------|----------------------------------------------------------------------|
| CHANGED                     | status CHANGED                                                       |
| CHANGED TO                  | status CHANGED TO "Done"                                             |
| CHANGED FROM                | status CHANGED FROM "Open"                                           |
| CHANGED FROM x TO y         | status CHANGED FROM "In Progress" TO "Done"                         |
| CHANGED BY "user"           | status CHANGED BY "john.doe"                                         |
| CHANGED AFTER "date"        | status CHANGED TO "Done" AFTER "2024-01-01"                          |
| CHANGED BEFORE "date"       | status CHANGED BEFORE "-7d"                                          |
| CHANGED DURING ("d1","d2")  | status CHANGED DURING ("2024-01-01","2024-03-31")  ← only valid use of DURING |
| WAS "value"                 | status WAS "In Progress"                                             |
| WAS IN (...)                | status WAS IN ("In Progress", "In Review")                           |
| WAS NOT IN (...)            | status WAS NOT IN ("Closed", "Done")                                 |
| WAS "x" BY "user"           | status WAS "Open" BY "john.doe"                                      |
| WAS "x" BEFORE "date"       | status WAS "Open" BEFORE "-30d"                                      |
| WAS "x" AFTER "date"        | status WAS "Done" AFTER "2024-01-01"                                 |
| NOT CHANGED                 | priority NOT CHANGED                                                 |

### Date Functions
✓ = Jira Server v9 supported   ✗ = Cloud only — never use on Server

| Function              | Server | Description                                      |
|-----------------------|--------|--------------------------------------------------|
| startOfDay()          | ✓      | Midnight today                                   |
| endOfDay()            | ✓      | End of today                                     |
| startOfWeek()         | ✓      | Monday of current week                           |
| endOfWeek()           | ✓      | Sunday of current week                           |
| startOfMonth()        | ✓      | First day of current month                       |
| endOfMonth()          | ✓      | Last day of current month                        |
| startOfYear()         | ✓      | Jan 1 of current year                            |
| endOfYear()           | ✓      | Dec 31 of current year                           |
| startOfQuarter()      | ✗      | Cloud only — DO NOT USE on Jira Server           |
| endOfQuarter()        | ✗      | Cloud only — DO NOT USE on Jira Server           |
| startOfDay("-1")      | ✓      | Yesterday midnight (offset supported by all ✓)   |
| startOfMonth("+1")    | ✓      | First day of next month                          |
| "-7d" / "-2w" / "-1M" | ✓      | Relative offset string (days/weeks/months)       |
| "2024-01-15"          | ✓      | Absolute date literal                            |

Quarter alternative for Jira Server (compute start date from today and use absolute literal):
  Q1: created >= "YYYY-01-01" AND created <= "YYYY-03-31"
  Q2: created >= "YYYY-04-01" AND created <= "YYYY-06-30"
  Q3: created >= "YYYY-07-01" AND created <= "YYYY-09-30"
  Q4: created >= "YYYY-10-01" AND created <= "YYYY-12-31"
  "This year" alternative: created >= startOfYear()

### User & Membership Functions
| Function            | Example                                              |
|---------------------|------------------------------------------------------|
| currentUser()       | assignee = currentUser()                             |
| membersOf("group")  | assignee in membersOf("jira-developers")             |

### Version Functions
| Function                       | Example                                                       |
|--------------------------------|---------------------------------------------------------------|
| releasedVersions("PROJ")       | fixVersion in releasedVersions("KAFKA")                       |
| unreleasedVersions("PROJ")     | fixVersion in unreleasedVersions("KAFKA")                     |
| latestReleasedVersion("PROJ")  | fixVersion = latestReleasedVersion("KAFKA")                   |
| earliestUnreleasedVersion("PROJ") | fixVersion = earliestUnreleasedVersion("KAFKA")            |

### Common Fields
| Field           | Notes                                                         |
|-----------------|---------------------------------------------------------------|
| project         | project = KAFKA  or  project in (KAFKA, SPARK)               |
| issuetype       | Bug, Story, Task, Epic, Sub-task, Improvement, New Feature    |
| status          | Open, In Progress, In Review, Resolved, Closed, Done, Reopened|
| priority        | Blocker, Critical, Major, Minor, Trivial                      |
| assignee        | username string or currentUser() or EMPTY                     |
| reporter        | username string or currentUser()                              |
| created         | Date issue was created                                        |
| updated         | Date of last update                                           |
| resolutiondate  | Date resolved — is EMPTY means unresolved                     |
| duedate         | Due date — duedate < now() finds overdue                      |
| fixVersion      | Target release version                                        |
| affectedVersion | Version the bug was found in                                  |
| component       | Component name                                                |
| labels          | Comma-separated label values                                  |
| sprint          | Sprint name — requires Agile plugin on Jira Server            |
| comment         | comment ~ "deployment" searches comment text                  |
| votes           | votes > 10 finds popular issues                               |
| watchers        | watchers = currentUser()                                      |
| parent          | parent = KAFKA-100 finds sub-tasks of an issue                |
| "Epic Link"     | "Epic Link" = KAFKA-50 finds issues under an epic             |
| text            | text ~ "kafka broker" searches summary+description+comments   |

### Quoting Rules
- Values with spaces MUST be quoted: status in (Open, "In Progress", "In Review")
- Single-word values do not need quotes: status = Open
- Always quote date literals: created >= "2024-01-01"
- Always quote usernames with dots or special chars: assignee = "john.doe"

### ORDER BY
Supported: created, updated, priority, status, assignee, reporter, summary, key, duedate, votes, watchers
Direction: ASC or DESC only.
Multiple: ORDER BY priority DESC, created ASC

### What Does NOT Exist in JQL — Never Generate These
- LIMIT       → use maxResults in params
- GROUP BY    → no aggregation in JQL; do it in post_processing
- HAVING      → does not exist
- COUNT()     → does not exist
- RANDOM()    → use ORDER BY updated DESC with small maxResults
- DURING      → only valid as predicate after CHANGED/WAS, NOT as standalone date range operator
- JOIN        → does not exist; use multiple conditions instead
- SUBQUERY    → does not exist
- Feature     → not a valid issuetype on most instances; prefer "New Feature" or omit

### Complex Query Patterns

**Overdue open issues:**
  project = KAFKA AND duedate < now() AND status not in (Done, Closed, Resolved)

**Issues reopened after being closed:**
  project = KAFKA AND status CHANGED FROM "Closed" TO "Reopened"

**Issues that were In Progress but never reached Done:**
  project = KAFKA AND status WAS "In Progress" AND status != Done

**Issues moved to Done by a specific person this week:**
  project = KAFKA AND status CHANGED TO "Done" BY "john.doe" AFTER startOfWeek()

**Issues stuck in same status for over 7 days:**
  project = KAFKA AND status != Done AND status CHANGED BEFORE "-7d"

**Unresolved Blockers or Criticals created this quarter (Server-safe):**
  project = KAFKA AND priority in (Blocker, Critical) AND resolutiondate is EMPTY AND created >= startOfYear()

**Bug count by priority (fetch all, group in post_processing):**
  project = KAFKA AND issuetype = Bug AND status not in (Done, Closed)
  post_processing: count issues grouped by priority field

**Cycle-time analysis — top N slowest to close:**
  project = KAFKA AND resolutiondate is not EMPTY ORDER BY created ASC
  maxResults: 100
  post_processing: compute (resolutiondate - created) days per issue, sort DESC, take top N

**Workload distribution — issues per assignee:**
  project = KAFKA AND status not in (Done, Closed) AND assignee is not EMPTY
  post_processing: group by assignee, count issues per person

**Sprint burndown — resolved vs open this sprint:**
  project = KAFKA AND sprint = "Sprint 42"
  post_processing: count issues by status category (done vs not done)

**Issues that spent time in a specific status (changelog required):**
  project = KAFKA AND status WAS "In Review"
  needs_changelog: true
  post_processing: for each issue compute duration in "In Review" from changelog transitions

**Issues fixed in the latest release:**
  project = KAFKA AND fixVersion = latestReleasedVersion("KAFKA") AND status = Closed

**Issues with no activity in last N days (stale):**
  project = KAFKA AND status not in (Done, Closed) AND updated <= "-10d" ORDER BY updated ASC

**Issues with many votes (community demand):**
  project = KAFKA AND votes > 20 AND status not in (Done, Closed) ORDER BY votes DESC

**Multi-project query:**
  project in (KAFKA, SPARK, FLINK) AND priority = Blocker AND status != Closed

**Text search across summary, description, comments:**
  project = KAFKA AND text ~ "consumer group rebalance"

### Fetching Top N by Computed Metric
JQL cannot sort by computed values. Always:
1. Fetch a larger set (50–100 resolved issues) via JQL
2. Compute the metric per issue in post_processing
3. Sort and take top N in post_processing
"""
