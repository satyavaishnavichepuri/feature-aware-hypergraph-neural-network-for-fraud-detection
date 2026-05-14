# contributing to feature-aware hypergraph neural network for fraud detection

thanks for your interest in contributing! this guide will help you get started.

## getting started

1. **choose the issue**
2. **fork the repository** and clone it locally
3. **set up your environment**:
   ```bash
   pip install torch torch_geometric scikit-learn pandas numpy matplotlib
   ```

## project structure

```
.
├── fa2b.py                    # main pipeline code
├── mock_data/                 # lightweight test datasets
├── tests/                     # unit tests
├── plots/                     # generated visualizations
├── docs/                      # documentation
└── README.md
```

## types of contributions

we welcome various types of contributions:

- **documentation** (#10): add docstrings, improve README, write tutorials
- **testing** (#9): add unit tests for pipeline components
- **enhancements** (#2, #4, #5, #6, #7, #8): improve existing features
- **deployment** (#6): API development, model serving
- **data** (#3): create mock datasets for testing
- **code quality** (#1): improve readability and maintainability

look for issues labeled:
- `good first issue` - beginner-friendly tasks
- `help wanted` - we especially need help here
- `documentation` - documentation improvements
- `enhancement` - feature additions or improvements

## how to contribute

### 1. pick an issue

- browse open issues and find one that interests you
- comment on the issue: "i'd like to work on this"
- wait for me to assign it to you before starting work
- if you have questions about the issue, ask in the comments

### 2. understand the codebase

**key components:**
- `load_dataset()` - loads the three CSV files (features, classes, edges)
- `preprocess()` - handles scaling, normalization, label mapping
- `build_all_clusters()` - creates hyperedges via clustering
- `prune_p75()` - filters low-quality hyperedges
- `build_graph()` - constructs PyTorch Geometric graph
- `temporal_split()` - splits data by timesteps
- `train()` - trains the GCN model
- `evaluate()` - computes metrics and breakdowns

**data format:**
the pipeline expects three CSV files:

1. **features file**: transaction_id, timestep, feature_1, ..., feature_n
2. **classes file**: transaction_id, class (1=illicit, 2=licit, unknown/3=unlabeled)
3. **edgelist file**: source_transaction_id, target_transaction_id

### 3. create a branch

create a descriptive branch name:
```bash
git checkout -b type/issue-number-short-description
```

examples:
- `docs/10-add-function-docstrings`
- `test/9-add-unit-tests`
- `feat/5-alternative-graph-models`
- `fix/2-hyperparameter-tuning`

### 4. make your changes

**code style:**
- use lowercase with underscores for function names: `load_dataset()`, `build_graph()`
- add comments for complex logic
- keep functions focused and single-purpose
- follow existing patterns in the codebase

**documentation style:**
- use lowercase except for proper nouns (PyTorch, Elliptic, etc.)
- be clear and concise
- include examples where helpful
- explain the "why" not just the "what"

**for testing contributions:**
- place test files in `tests/` directory
- name test files `test_*.py`
- use pytest framework
- aim for >80% coverage of critical paths

### 5. commit your changes

**commit message format:**
```
type(scope): brief description (#issue-number)

optional longer description explaining why this change
was made and any important context
```

**types:**
- `feat`: new feature
- `fix`: bug fix
- `docs`: documentation only
- `test`: adding or updating tests
- `refactor`: code refactoring
- `perf`: performance improvement
- `style`: formatting, no code change

**examples:**
```bash
git commit -m "docs(functions): add docstrings for clustering functions (#10)"
git commit -m "test(pipeline): add unit tests for preprocessing (#9)"
git commit -m "feat(model): experiment with GAT architecture (#5)"
```

### 6. push and create pull request

```bash
git push origin your-branch-name
```

**pull request format:**

**title:** `type: brief description (#issue-number)`

**description template:**
```markdown
## description
brief explanation of what this PR does

## related issue
closes #[issue-number]

## changes made
- bullet point list of specific changes
- be concrete and specific

## testing performed
- how you tested your changes
- what scenarios you covered
- any edge cases considered

## checklist
- [ ] code follows project style
- [ ] added/updated tests (if applicable)
- [ ] added/updated documentation
- [ ] tested locally
- [ ] referenced issue number
```

**example PR:**
```markdown
## description
added comprehensive docstrings to all clustering and hyperedge construction functions

## related issue
closes #10

## changes made
- added docstrings to build_all_clusters() explaining parameters and return values
- documented the clustering algorithm logic
- added examples in docstrings showing expected input/output
- updated inline comments for clarity

## testing performed
- verified docstrings render correctly in IDE tooltips
- checked that examples in docstrings are accurate
- no functional code changes, only documentation

## checklist
- [x] code follows project style
- [x] added/updated tests (if applicable)
- [x] added/updated documentation
- [x] tested locally
- [x] referenced issue number
```

## testing expectations

**for code changes:**
- test your changes locally before submitting PR
- if adding new functions, include example usage
- if modifying existing functions, verify backwards compatibility

**for documentation:**
- ensure formatting renders correctly
- verify all code examples run without errors
- check for typos and clarity

**for data contributions:**
- validate CSV format matches expected structure
- verify the pipeline can load your data
- include sample output or screenshots if helpful

## code review process

1. **submission**: you open a PR following the format above
2. **automated checks**: none yet, manual review only
3. **my review**: i'll review within 1-3 days and provide feedback
4. **validation**: i'll run the pipeline on your contribution and attach results
5. **iteration**: you address any requested changes
6. **approval**: once everything looks good, i'll approve
7. **merge**: your contribution gets merged!

**what i look for:**
- does it solve the issue completely?
- is the code clear and maintainable?
- are there any edge cases to consider?
- does it follow existing patterns?
- is documentation clear?

**typical review timeline:**
- initial review: 1-3 days
- follow-up reviews: 1-2 days
- total time to merge: usually 3-7 days depending on complexity

## asking for help

- **questions about an issue?** comment on the issue thread
- **stuck on something?** open a discussion or comment on your PR
- **found a bug?** open a new issue with details

## resources

- **original elliptic dataset**: https://www.kaggle.com/datasets/ellipticco/elliptic-data-set
- **pytorch geometric docs**: https://pytorch-geometric.readthedocs.io/
- **bitcoin transaction graphs**: for understanding the domain

## code of conduct

- be respectful and professional
- help others learn and grow
- constructive feedback only
- no spam or off-topic content

---

thank you for contributing! every contribution, no matter how small, helps improve this project.
