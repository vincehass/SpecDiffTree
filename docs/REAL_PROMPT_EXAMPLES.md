# Real Evaluation Prompts - Examples

This document shows what the **ACTUAL evaluation prompts** look like for each dataset.

---

## M4 Time Series Captioning (Stage 2)

### Example Prompt to Model:

```
You are an expert in time series analysis.This is the time series, it has mean 8103.0078 and std 2421.9387:Please generate a detailed caption for this time-series, describing it as accurately as possible.
```

**Length**: ~200 characters, ~30 words

###Expected Output:

```
The time-series graph illustrates data points over a span of approximately 160 units on the x-axis,
with values ranging from 6000 to 15000 on the y-axis. The trend begins with a relatively stable phase,
followed by a gradual increase starting around the 30th unit. This upward trajectory becomes more
pronounced after the 50th unit, peaking sharply near the 80th unit at a value exceeding 14000.
Subsequently, a decline is observed, with fluctuations around the 100th unit, before a more consistent
decline toward the end of the series...
```

**Length**: ~760 characters, ~118 words

---

## HAR Activity Recognition with Chain-of-Thought (Stage 3)

### Example Prompt to Model:

```
You are given accelerometer data in all three dimensions. Your task is to classify the activity based on analysis of the data.

Instructions:
- Begin by analyzing the time series without assuming a specific label.
- Think step-by-step about what the observed patterns suggest regarding movement intensity and behavior.
- Write your rationale as a single, natural paragraph — do not use bullet points, numbered steps, or section headings.
- Do **not** mention any class label until the final sentence.

Possible activity labels are:
biking, lying, running, sitting, standing, walking, walking_down, walking_up.

- Make sure that your last word is the answer. You MUST end your response with "Answer: "

The following is the accelerometer data on the x-axis, it has mean -3.2434 and std 0.0474:
The following is the accelerometer data on the y-axis, it has mean 2.3132 and std 0.0550:
The following is the accelerometer data on the z-axis, it has mean 8.9963 and std 0.0400:
```

**Length**: ~1050 characters, ~157 words

### Expected Output:

```
The accelerometer data over the 2.56 second window shows relatively low variability and consistent
patterns across the X, Y, and Z axes. The X-axis data fluctuates within a narrow range, indicating
minimal lateral movement. Similarly, the Y-axis maintains a stable pattern with slight variations,
suggesting limited vertical movement. The Z-axis, which often reflects the primary force of gravity,
also displays minor fluctuations, pointing to a stable orientation. These characteristics collectively
indicate a state of minimal physical activity, with no significant dynamic movements or position changes.
The overall pattern suggests a stationary or resting position where the body is not actively engaged
in movement.

Answer: lying
```

**Length**: ~780 characters, ~108 words

---

## Key Observations

### 1. **Prompts Are Complex**

- M4: Requires understanding statistical summaries (mean, std) and generating descriptive captions
- HAR: Requires analyzing 3-axis accelerometer data and providing chain-of-thought reasoning

### 2. **Real Data Included**

- M4: Mean and standard deviation of actual time series
- HAR: Mean and std for each axis (x, y, z)

### 3. **Expected Outputs Are Sophisticated**

- Not just single words or short answers
- Require coherent paragraphs with logical reasoning
- HAR specifically requires step-by-step analysis before final classification

### 4. **Model Challenges**

- Must understand domain-specific concepts (time series trends, accelerometer physics)
- Must generate fluent, structured text
- Must follow specific formatting requirements (e.g., "Answer: X" at the end)
- Must reason through evidence before concluding

---

## Why This Matters for Evaluation

The previous evaluation with **empty templates** was not meaningful because:

- ❌ No actual data in prompts
- ❌ Model had nothing to analyze
- ❌ Outputs were random/meaningless

With **real prompts** from datasets:

- ✅ Model receives actual data to analyze
- ✅ Prompts test understanding of domain concepts
- ✅ Outputs can be compared to ground truth
- ✅ Evaluation measures actual capability

---

## Next Steps

To run meaningful evaluation:

1. **Use real datasets** (`M4QADataset`, `HARCoTQADataset`)
2. **Extract complete prompts** (pre_prompt + time_series_text + post_prompt)
3. **Pass to MaxEnt-TS** tree search
4. **Compare outputs** to ground truth answers
5. **Measure quality** (BLEU, ROUGE, exact match, etc.)

This will give us **real metrics** showing whether:

- The model understands the data
- MaxEnt-TS improves generation quality
- Spectral rewards help with time series tasks
