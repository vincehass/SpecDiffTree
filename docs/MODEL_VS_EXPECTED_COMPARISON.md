# Model Output vs. Expected Output Comparison

This document shows what the model **actually generates** vs. what it **should generate** (ground truth).

---

## 📊 Test 1: M4 Time Series Captioning

### Input Prompt:

```
You are an expert in time series analysis.
This is the time series, it has mean 8103.0078 and std 2421.9387:
Please generate a detailed caption for this time-series, describing it as accurately as possible.
```

### 🤖 ACTUAL MODEL OUTPUT:

```


Please provide a brief description of the time series.

Brief Description: A time series has been created with an average value of 8103.0078
and a standard deviation of 2421.9387. This pattern can be analyzed using various time
series analysis techniques. A detailed description of the time series should be provided.

## Step 1: Identify the nature of the time series
Given the mean and standard deviation of a time series, we can first estimate the
complexity of the series. A time series with a clear pattern or trend, a high degree
of volatility (high standard deviation), or both would be classified as a complex series.

## Step 2: Describe the time series
The time series appears to be a continuous, non-stationary series, as indicated by its
non-zero mean (8103.0078) and standard deviation (2421.9387). This suggests that the
mean of the series might have changed over time, and the variability in the data
```

**Length:** 919 characters, 152 words  
**Word Diversity:** 55.92%

### 🎯 EXPECTED OUTPUT (Ground Truth):

```
The time-series graph illustrates data points over a span of approximately 160 units
on the x-axis, with values ranging from 6000 to 15000 on the y-axis. The trend begins
with a relatively stable phase, followed by a gradual increase starting around the 30th
unit. This upward trajectory becomes more pronounced after the 50th unit, peaking sharply
near the 80th unit at a value exceeding 14000. Subsequently, a decline is observed, with
fluctuations around the 100th unit, before a more consistent descent continues until the
end of the series. Overall, the time-series exhibits a clear peak and subsequent decline,
indicating a possible event or trend reversal occurring in the middle portion of the data.
```

**Length:** ~760 characters

### 📈 Comparison:

| Aspect       | Model Output                    | Expected Output                |
| ------------ | ------------------------------- | ------------------------------ |
| **Style**    | Structured with steps           | Narrative description          |
| **Content**  | Generic analysis methodology    | Specific visual description    |
| **Details**  | Mentions mean/std, stationarity | Describes peaks, units, values |
| **Accuracy** | No visual specifics             | Timeline-based description     |

---

## 🏃 Test 2: HAR Activity Recognition (Chain-of-Thought)

### Input Prompt:

```
You are given accelerometer data in all three dimensions. Your task is to classify
the activity based on analysis of the data.

Instructions:
- Begin by analyzing the time series without assuming a specific label.
- Think step-by-step about what the observed patterns suggest regarding movement
  intensity and behavior.
- Write your rationale as a single, natural paragraph.
- Do **not** mention a specific activity label in your rationale.
- Then, after providing your detailed analysis, give your final answer in the format:
  Answer: [LAYING, SITTING, STANDING, WALKING, WALKING_DOWNSTAIRS, WALKING_UPSTAIRS]

The following is the accelerometer data on the x-axis, it has mean -3.2434 and std 0.0474:
The following is the accelerometer data on the y-axis, it has mean 2.3132 and std 0.0550:
The following is the accelerometer data on the z-axis, it has mean 9.2017 and std 0.0639:

Rationale:
```

### 🤖 ACTUAL MODEL OUTPUT:

```
 A strong correlation between the three axes suggests this person is highly active,
with a sense of excitement and possibly a slight increase in body temperature, the
x and y values are larger than the z-axis indicating a general increase in movement
intensity, and the standard deviations indicate a fairly stable activity level while
still showing a significant increase in activity."

        Answer: The following is the accelerometer data on the x-axis, it has mean
-3.2434 and std 0.0474: The following is the accelerometer data on the y-axis, it
has mean 2.3132 and std 0.0550: The following is the accelerometer data on the z-axis,
it has mean 9.2017 and std 0.0639:Rationale: A strong correlation between the three
axes suggests this person is highly active, with a sense of excitement and possibly
a slight increase in body temperature, the x and y values are larger than the z-axis
indicating a general increase in movement intensity,
```

**Length:** 945 characters, 153 words  
**Word Diversity:** 43.79%

### 🎯 EXPECTED OUTPUT (Ground Truth):

```
The accelerometer data over the 2.56 second window shows relatively low variability
and consistent patterns across the X, Y, and Z axes. The X-axis data fluctuates within
a narrow range, indicating minimal lateral movement. Similarly, the Y-axis maintains
a stable pattern with slight variations, suggesting limited vertical movement. The Z-axis,
which often reflects the primary force of gravity, also displays minor fluctuations,
pointing to a stable orientation. These characteristics collectively suggest a stationary
or minimally active state where the subject is not engaged in significant physical
movement. Given the low standard deviations and the mean values that suggest gravity
is primarily acting along the Z-axis, the subject is most likely upright and motionless,
typical of a standing posture.

Answer: STANDING
```

**Length:** ~780 characters

### 📈 Comparison:

| Aspect                 | Model Output                       | Expected Output                 |
| ---------------------- | ---------------------------------- | ------------------------------- |
| **Activity Analysis**  | "highly active", "excitement"      | "low variability", "stationary" |
| **Movement Intensity** | "significant increase in activity" | "minimal lateral movement"      |
| **Reasoning**          | Incorrect interpretation of data   | Correct: low std = stable       |
| **Final Answer**       | No clear classification            | **STANDING**                    |
| **Accuracy**           | ❌ WRONG                           | ✅ CORRECT                      |

---

## 🔍 Key Problems with Baseline Model:

### Problem 1: M4 Captioning

- **Model says:** Generic methodology about analyzing statistics
- **Should say:** Specific visual description of trends and peaks
- **Issue:** Model doesn't "see" the actual time series pattern

### Problem 2: HAR Classification

- **Model says:** "highly active", "excitement", "increase in activity"
- **Should say:** "low variability", "stationary", "STANDING"
- **Issue:** Model misinterprets low standard deviations as high activity (OPPOSITE!)

---

## 💡 Why MaxEnt-TS Should Help:

1. **Tree Search:** Explores multiple output possibilities instead of greedy decoding
2. **Spectral Reward:** Guides the model toward outputs that preserve frequency content
3. **Best Path Selection:** Chooses the completion with highest reward

**Next Step:** Run full evaluation with MaxEnt-TS to see if tree search improves these outputs!

---

## 📁 Files to Check:

- **This comparison:** `MODEL_VS_EXPECTED_COMPARISON.md`
- **Test script:** `test_model_outputs.py`
- **Evaluation results:** `evaluation/results/stage2_pytorch.json` and `stage3_pytorch.json`

The evaluation results JSON files will contain arrays of samples with:

- `prompt`: The input
- `generated_text`: What the model produced
- `ground_truth`: What it should have produced
- `reward`: Quality score from spectral reward function
