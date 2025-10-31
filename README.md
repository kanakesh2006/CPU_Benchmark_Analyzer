
---

# **⚙️ CPU Benchmark Analyzer**

### “Measure. Analyze. Visualize your CPU Performance.”

---

## 🧩 **Project Overview**

**CPU Benchmark Analyzer** is a **Python-based GUI application** designed to test and visualize the **performance of a computer’s processor (CPU)** through various computational benchmarks.
It offers users an **interactive, modern, and visually appealing interface** to evaluate their CPU’s efficiency in performing different tasks — from arithmetic operations to data processing and matrix computations.

This project demonstrates the **practical application of Computer Organization & Architecture concepts**, such as CPU performance, ALU and FPU operations, multi-core utilization, and instruction execution time.

---

## 🎯 **Objective**

* To **analyze and compare CPU performance** using real-time benchmark tests.
* To **visualize performance metrics** through modern GUI and graphical representations.
* To provide a **user-friendly interface** for observing processor characteristics and performance variations.

---

## 🖥️ **Technology Stack**

| Component             | Technology                    | Purpose                         |
| --------------------- | ----------------------------- | ------------------------------- |
| GUI Framework         | **Tkinter**                   | Interactive front-end interface |
| System Monitoring     | **psutil, cpuinfo, platform** | Fetch CPU details               |
| Benchmark Computation | **NumPy, math, time**         | Perform CPU-intensive tasks     |
| Data Visualization    | **Matplotlib**                | Plot benchmark results          |
| Language              | **Python 3.x**                | Core application logic          |

---

## 🚀 **Key Features**

### 🧾 1. **Real-Time CPU Information**

* Fetches CPU brand, architecture, core and thread count, base frequency, and logical cores.
* Displays system specifications instantly through a single click.

### 🧮 2. **Multiple Benchmark Tests**

Performs a variety of **real computational benchmarks**, including:

| Benchmark Type                   | Description                                           | What It Tests                             |
| -------------------------------- | ----------------------------------------------------- | ----------------------------------------- |
| **Integer Arithmetic Test**      | Performs millions of integer operations               | ALU performance                           |
| **Floating-Point Test**          | Calculates large numbers of floating-point operations | FPU efficiency                            |
| **Matrix Multiplication**        | Uses NumPy to multiply large matrices                 | Linear algebra and memory bandwidth       |
| **Sorting Benchmark**            | Sorts a large list of random values                   | Algorithmic and data handling performance |
| **Prime Number Test (optional)** | Finds all primes up to N                              | Loop + computation efficiency             |
| **Fibonacci Test (optional)**    | Generates Fibonacci series recursively                | Recursive operation efficiency            |

### 📊 3. **Performance Visualization**

* Uses **Matplotlib** to generate bar charts comparing benchmark times.
* Displays CPU performance metrics in an **easy-to-understand graphical format**.

### 🎨 4. **Modern & Interactive GUI**

* Intuitive interface with clean color themes and structured layout.
* Buttons, text boxes, and result sections organized professionally.
* Real-time status messages (“Running Benchmarks…”, “Completed Successfully ✅”).
* Attractive title banner and labels.

### 💾 5. **Result Summary**

* Displays total time taken by each benchmark.
* Option to view test scores side-by-side for comparison.
* Performance ratings based on test outcomes (e.g., Excellent, Average, Slow).

---

## 🧠 **How It Works**

1. **Launch the Application**
   Run `python main.py` to open the GUI.

2. **View System Information**
   Click **“Show CPU Info”** to see processor details fetched via `cpuinfo` and `psutil`.

3. **Run Benchmarks**
   Click **“Run Benchmarks”** to execute multiple tests — arithmetic, floating-point, sorting, matrix multiplication, etc.

4. **View Results**
   Benchmark times are displayed in the text area below.

5. **Visualize Data**
   Click **“Show Graph”** to generate a performance comparison graph using Matplotlib.

---

## 📈 **Sample Output (Text Display)**

```
Running benchmarks...

✅ Benchmark Results:
Integer Test: 1.124 seconds
Floating-Point Test: 0.842 seconds
Matrix Multiplication: 2.315 seconds
Sorting Test: 1.597 seconds
```

---

## 📊 **Sample Output (Graph)**

A bar graph showing benchmark comparison:

```
|                 Benchmark Comparison                 |
|-------------------------------------------------------|
| Integer Test         ████████████████  1.12s          |
| Floating-Point Test  ████████████      0.84s          |
| Matrix Test          █████████████████████████ 2.31s  |
| Sorting Test         ███████████████   1.59s          |
```

---

## 🧰 **Installation & Setup**

### 1️⃣ Install Dependencies

```bash
pip install psutil py-cpuinfo numpy matplotlib
```

### 2️⃣ Run the Application

```bash
python main.py
```

### 3️⃣ Interact with GUI

* “Show CPU Info” → View hardware details
* “Run Benchmarks” → Start performance tests
* “Show Graph” → Visualize results

---

## 🧪 **Concepts Demonstrated (COA Linkage)**

| Concept                           | Demonstration in App                  |
| --------------------------------- | ------------------------------------- |
| **CPU Architecture**              | Shows processor type and specs        |
| **ALU & FPU Operations**          | Integer and floating-point benchmarks |
| **Instruction Execution Time**    | Benchmark duration measurement        |
| **Parallelism & Cores**           | Observed via faster multi-core CPUs   |
| **System Performance Evaluation** | Through comparative test results      |

---

## 🧑‍💻 **About & Credits**

**Project Title:** CPU Benchmark Analyzer

**Subject:** Computer Organization and Architecture

**Developed Using:** Python (Tkinter, NumPy, psutil, Matplotlib)

**Developer:** Kanakesh

**Institution:** SRM Institute of Science and Technology

**Academic Year:** 2025–26

---

## 🌟 **Tagline**

> “Turning CPU power into measurable performance.”

> “Benchmark your processor. Visualize your machine’s true potential.”

---

All rights reserved by Kanakesh Kapaganti.

