# Neural Network Assignment Notebook

This notebook consists of **three main parts**, each building on the previous one to implement and test a fully working feedforward neural network from scratch.  
You’ll start with the foundations of a logistic layer, then extend it into a multi-layer network with proper backpropagation, and finally train and evaluate it on several datasets.

---

## 🧩 Part 1 — Logistic Layer Gradients

**Goal:**  
Adapt the `LogisticLayer` to the full neural-network setting by implementing correct gradient formulas for backprop through a logistic (sigmoid) layer.

### What to Implement
- Keep the **forward pass**:
  \[
  Z = A_{\text{prev}} W^\top + b, \quad A = g(Z)
  \]
  where \(g(Z)\) is the sigmoid activation.

- Implement the **backward pass** for a batch:
  - Upstream gradient: \(\frac{\partial \ell}{\partial A}\)
  - Local gradients:
    \[
    g'(Z) = A \odot (1 - A)
    \]
    \[
    \frac{\partial \ell}{\partial Z} = \frac{\partial \ell}{\partial A} \odot g'(Z)
    \]
    \[
    \frac{\partial \ell}{\partial W} = (\frac{\partial \ell}{\partial Z})^\top A_{\text{prev}}
    \]
    \[
    \frac{\partial \ell}{\partial b} = \text{sum over samples}(\frac{\partial \ell}{\partial Z})
    \]
    \[
    \frac{\partial \ell}{\partial A_{\text{prev}}} = \frac{\partial \ell}{\partial Z} \, W
    \]
- Be careful with **batch shapes** and **broadcasting** for \(b\).

### Tests (Q1)
Add or modify tests to verify:
- All gradients have correct shapes.
- Gradients aren’t all zeros or NaNs.
- Numerical gradient check (finite differences) matches your analytic gradients.

---

## 🔁 Part 2 — Backpropagation in the `NN` Class

**Goal:**  
Implement end-to-end backprop in the `NN` class by chaining layer backward calls and computing the loss gradient at the output.

### What to Implement
- Compute the gradient of the **binary cross-entropy** loss w.r.t. predictions \(\hat{Y}\):
  \[
  \frac{\partial \ell}{\partial \hat{Y}} = -\,\frac{Y}{\hat{Y}} + \frac{1-Y}{1-\hat{Y}}
  \]
- Run **backward** through layers in reverse order:
  - Pass the upstream gradient into each layer’s `backward()` function.
  - Collect each layer’s gradients for \(W\) and \(b\).
- Update parameters using the provided optimizer or `step()` logic.

### Tests (Q2)
- Verify that after a backward-then-update step, the loss decreases on a small batch.
- Check that forward-cache values remain unchanged.
- (Optional) Run a numerical gradient check on the full model.

---

## 🚀 Part 3 — Training and Experiments

**Goal:**  
Train and evaluate your network on several datasets to verify it learns correctly.

### Case 1 — Logic Gates
- Provided example trains an **AND** gate; it should converge quickly if everything works.
- **Assignment 3:** Tune hyperparameters (hidden size, learning rate, epochs) and inspect error patterns.
- **Assignment 4:** Fix **XOR** training (requires a hidden layer and non-linear activation).

### Case 2 — Digits Dataset
- Use the `digits` dataset (from sklearn or provided cell).
- **Assignment 5:** Define and train a network `digit_NN` for this dataset.
- Target: **≥ 90% test accuracy**.
- Use `plot_costs` to visualize the loss curve.

### Case 3 — Iris Dataset
- **Assignment 6a:** Normalize features to the range [0, 1].
- **Assignment 6b:** Convert class labels to **one-hot encoding**.
- **Assignment 6c:** Perform a **train/test split** (30% test set).
- **Assignment 6d:** Train, evaluate, and plot training cost progression.

---

## ✅ Expected Learning Outcomes
By completing this assignment, you should be able to:
- Derive and implement analytic gradients for logistic units.
- Build a functioning backpropagation pipeline.
- Train small feedforward networks on logical, numeric, and categorical datasets.
- Validate gradient correctness via numerical checks.
- Visualize and interpret learning behavior across epochs.

---

## 📎 Notes
- Each part builds directly on the previous one — do not skip cells marked as **TODO**.
- Check intermediate outputs frequently using assertions and gradient norms.
- The final notebook should demonstrate working training runs for **AND**, **XOR**, **digits**, and **iris** datasets.

---
