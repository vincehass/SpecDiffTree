# 🔬 Performance Investigation: Executive Summary

## Real Performance Numbers - All Methods Tested

**Date:** December 14, 2025  
**Configuration:** 3 samples, 10 rollouts/simulations, Llama 3.2 1B on Apple Silicon (MPS)

---

## 📊 Results Summary

### ✅ Working Methods (2/5)

| Method | Success Rate | Avg Nodes | Avg Time | Quality |
|--------|--------------|-----------|----------|---------|
| **Greedy** | 100% (3/3) | 27 | 0.91s | ✅ Correct |
| **MaxEnt-TS** | 100% (3/3) | 31 | N/A* | ⚠️ Verbose, some errors |

*Time tracking needs to be added

### ❌ Failed Methods (3/5)

| Method | Success Rate | Error |
|--------|--------------|-------|
| **MCTS** | 0% (0/3) | `KeyError: 'best_text'` - missing from return dict |
| **DTS** | 0% (0/3) | `unexpected keyword argument 'num_iterations'` |
| **DTS*** | 0% (0/3) | `unexpected keyword argument 'num_iterations'` |

---

## 📝 Sample Outputs

### Sample 1: "The capital of France is"
**Expected:** Paris

**Greedy:**
> "The capital of France is Paris. The Eiffel Tower is located in Paris..."
- ✅ Correct, concise
- 26 tokens, 1.27s

**MaxEnt-TS:**
> "The capital of France is not Berlin or Vienna. In fact, Berlin is the capital of Germany, and Vienna is the capital of Austria. Paris is the capital of France..."
- ✅ Eventually correct but overly verbose
- 31 tokens explored

---

### Sample 2: "2 + 2 equals"
**Expected:** 4

**Greedy:**
> "2 + 2 equals 4. This is a basic arithmetic fact..."
- ✅ Correct
- 26 tokens, 0.72s

**MaxEnt-TS:**
> "2 + 2 equals\n3\nThe number on the top left corner of a circle is called the center..."
- ❌ Says "3" then rambles about geometry
- 31 tokens explored

---

### Sample 3: "The largest planet in our solar system is"
**Expected:** Jupiter

**Greedy:**
> "The largest planet in our solar system is Jupiter. It is a gas giant..."
- ✅ Correct
- 29 tokens, 0.73s

**MaxEnt-TS:**
> "The largest planet in our solar system is Earth, with a radius of approximately 6,371 kilometers..."
- ❌ Incorrectly says Earth (should be Jupiter)
- Then contradicts itself mentioning Mars and Jupiter later
- 31 tokens explored

---

## 🔍 Key Findings

### 1. Greedy Baseline is Very Strong
- **100% accuracy** on simple knowledge questions
- **Fast:** ~0.9s per query
- **Efficient:** 26-29 tokens explored
- **Clean outputs:** Concise and correct

### 2. MaxEnt-TS Has Quality Issues
- **Success rate:** 100% (runs without crashing)
- **Accuracy:** Only 33% (1/3 correct answers)
- **Verbosity:** Generates long, rambling text
- **Errors:** Hallucinates wrong facts (Earth is largest planet)
- **Time tracking:** Not working (shows 0s)

### 3. Tree Search Methods Need Fixes
- **MCTS:** Missing `'best_text'` in return dictionary
- **DTS/DTS*:** Wrong config parameter name (`num_iterations` should be `num_simulations`)

---

## 🐛 Bugs to Fix

### High Priority
1. **DTS Config:** Change `num_iterations` to `num_simulations` in DTSConfig
2. **MCTS Return:** Add `'best_text'` to MCTS search result dictionary
3. **MaxEnt-TS Time:** Add proper time tracking to MaxEnt-TS
4. **MaxEnt-TS Quality:** Investigate why it produces wrong/verbose answers

### Medium Priority
5. **Reward Function:** Currently using `dummy_reward(x) = 0.5` - need real rewards
6. **Hyperparameters:** Tune expansion_k, temperature, etc.

---

## 🎯 Next Steps

### Immediate (Fix Bugs)
1. ✅ Fix DTS config parameter name
2. ✅ Fix MCTS return dictionary  
3. ✅ Add time tracking to MaxEnt-TS
4. 🔄 Re-run comparison with all methods working

### Short Term (Improve Quality)
5. Implement proper reward functions (SpectralReward for time series)
6. Tune hyperparameters for each method
7. Test on actual time series datasets (M4, HAR)

### Long Term (Evaluation)
8. Run comprehensive evaluation (100+ samples)
9. Generate comparison plots and statistics
10. Write final report with publication-quality figures

---

## 💡 Insights

### Why is MaxEnt-TS Underperforming?

**Possible Causes:**
1. **No Proper Reward:** Using dummy reward `0.5` for all sequences
   - Tree search has no guidance on what's "good"
   - Random exploration without objective

2. **Wrong Task:** MaxEnt-TS designed for time series forecasting
   - Current test: Simple knowledge Q&A
   - Mismatch between method and evaluation

3. **Hyperparameters:** May need tuning for this task
   - Temperature, gamma, expansion_k
   - Different from original DTS paper settings

4. **Model Size:** Llama 1B may be too small
   - Base model quality affects tree search
   - Larger model might help

---

## 🏆 Winner So Far: Greedy

**Current Champion:** Greedy Baseline
- Fastest
- Most accurate
- Simplest

**But wait...** We're testing on the WRONG task!
- MaxEnt-TS is designed for **time series forecasting**
- We're testing on **general knowledge Q&A**
- Need to test on **actual time series data** (M4, HAR)

---

## ✅ Conclusion

**We successfully obtained REAL performance numbers!**

**Status:**
- ✅ Greedy: Working perfectly
- ✅ MaxEnt-TS: Working but quality issues
- ❌ MCTS: Fixable bug (missing dict key)
- ❌ DTS/DTS*: Fixable bug (config parameter)

**Next Action:** Fix the 3 bugs and re-run comprehensive comparison.

