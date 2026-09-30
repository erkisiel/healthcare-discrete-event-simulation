# Healthcare Discrete-Event Simulation

A **data-driven healthcare simulation project** developed for **IE 304** using Python and SimPy.

The project models the outpatient patient flow of a bone diseases hospital, including appointment-based and walk-in arrivals, registration, examination, X-ray processes, queues, resource utilization, and end-of-day behavior.

The model was designed to support scenario analysis and operational improvement decisions using discrete-event simulation and statistical analysis.

---

## Project Overview

The simulation represents a multi-clinic outpatient system with:

- Appointment-based patient arrivals
- Walk-in patients
- Seven clinical departments
- Registration desk
- X-ray registration
- Three X-ray rooms
- Primary examination
- Secondary/control examination
- Priority-based queues
- Lunch-break constraints
- End-of-day completion logic
- Stochastic service and arrival times

The implementation is built with **SimPy** and uses multiple independent random-number streams to support reproducible simulation experiments.

---

## Main Features

### Patient Flow

Patients may enter the system as either:

- **Scheduled patients**
- **Walk-in patients**

The model tracks patients through:

```text
Arrival
  ↓
Registration
  ↓
Clinical Examination
  ↓
X-Ray Decision
  ↓
X-Ray Registration
  ↓
X-Ray Room
  ↓
Return to Same Doctor
  ↓
Control Examination
  ↓
Exit
```

The project specification requires appointment schedules to be configurable and walk-in arrivals to follow an estimated distribution.

---

## Clinical Departments

The simulation includes seven clinical departments:

- Orthopedics 1
- Hand Surgery
- Pediatric Orthopedics
- Orthopedics 2
- Tumor
- Spine
- Sports Medicine

Each department is represented as a simulation resource and may use different patient schedules and X-ray probabilities.

---

## Appointment Schedules

The code supports multiple appointment schedules.

Examples include:

### Schedule 1
- Every 6 minutes from 08:00–12:00
- Every 6 minutes from 13:00–16:00

### Schedule 2
- Every 6 minutes from 08:00–10:00
- Every 10 minutes from 10:00–12:00
- Every 8 minutes from 13:00–15:00
- Every 10 minutes from 15:00–16:00

### Schedule 3
- Every 6 minutes from 08:00–10:00
- Every 8 minutes from 10:00–12:00

Patient arrival times are perturbed around their scheduled appointment times using a triangular distribution.

---

## Statistical Modeling

The simulation uses fitted probability distributions for service processes.

Examples include:

- Initial examination time
- Control examination time
- X-ray service time by room
- Walk-in arrivals
- Registration time
- Travel time between departments

Gamma distributions are used for several service-time processes.

---

## X-Ray Modeling

The X-ray subsystem includes:

- Department-specific probability of requiring X-ray
- Probability of choosing each X-ray room
- X-ray registration desk
- Three X-ray rooms
- X-ray service-time distributions
- Return flow to the original doctor
- Secondary examination after X-ray

The model also tracks X-ray utilization and queue lengths.

---

## Queueing Logic

The simulation uses SimPy resources and priority resources.

The code supports:

- Priority-based queues
- FIFO scenario testing
- Separate clinic queues
- Registration queues
- X-ray queues
- X-ray registration queue

The queue discipline can be changed during scenario experiments.

---

## Simulation Experiment Design

The base model runs:

- **30 replications**
- Independent seeded random streams
- Patient conservation checks
- End-of-day completion logic

For each replication, the model records system-level and resource-level metrics.

---

## Performance Metrics

The simulation evaluates:

- Average system time
- Average waiting time
- Maximum waiting time
- Clinic-level waiting times
- Queue lengths over time
- X-ray room utilization
- Registration queue length
- Number of patients sent to X-ray
- Number of late-arriving patients
- Number of patients remaining at end of day

---

## Statistical Validation

The project includes statistical analysis of simulation output.

### 95% Confidence Intervals

Confidence intervals are calculated for mean system time across replications.

### t-Test

A one-sample t-test compares the simulated mean system time against a reference value.

The analysis reports:

- Mean
- Standard deviation
- 95% confidence interval
- t-statistic
- p-value

---

## Design of Improvement

The project includes a structured scenario-analysis module.

Three main factors are evaluated:

1. **Walk-in patient ratio**
2. **Service-time multiplier**
3. **X-ray referral probability**

The model compares multiple Low / Medium / High combinations against the baseline system.

Example scenarios include:

- Reduced walk-in demand
- Increased walk-in demand
- Faster service times
- Slower service times
- Lower X-ray referral probability
- Higher X-ray referral probability
- Combined best-case scenario
- Combined worst-case scenario

Each scenario is evaluated across multiple replications.

---

## Sensitivity Analysis

A **One-at-a-Time (OAT)** sensitivity analysis is implemented for:

- Walk-in ratio
- Service-time scaling
- X-ray referral probability

The project generates confidence intervals and response plots showing how each parameter affects average system time.

---

## Visualizations

The model generates several analytical plots, including:

- System-time distribution
- Waiting-time distribution
- Clinic-level average waiting time
- X-ray room utilization
- Clinic queue-length time series
- X-ray queue-length time series
- Registration queue time series
- t-test visualization
- Scenario comparison charts
- Scenario boxplots
- Main-effects plots
- Improvement heatmaps
- Best / worst / baseline comparisons
- Sensitivity-analysis plots

---

## Technologies

- **Python**
- **SimPy**
- **NumPy**
- **Pandas**
- **Matplotlib**
- **Seaborn**
- **SciPy**
- Discrete-Event Simulation
- Queueing Systems
- Statistical Analysis
- Scenario Analysis
- Sensitivity Analysis

---

## Project Structure

```text
healthcare-discrete-event-simulation/
│
├── simulation.py
├── requirements.txt
├── README.md
│
├── docs/
│   └── System_for_project_2026.docx
│
└── results/
    └── generated simulation plots
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/erkisiel/healthcare-discrete-event-simulation.git
cd healthcare-discrete-event-simulation
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Simulation

Run:

```bash
python simulation.py
```

The simulation executes the base model, performs 30 replications, produces statistical summaries, and generates output figures.

Depending on the enabled analysis sections, it can also execute scenario and sensitivity experiments.

---

## Academic Context

This project was developed for an **Industrial Engineering simulation course (IE 304)**.

The system definition includes:

- Scheduled and walk-in arrivals
- Multiple hospital departments
- Department-specific X-ray probabilities
- Configurable appointment schedules
- Queue-discipline decisions
- Secondary screening by the same doctor
- Waiting-time and queue-length performance measures

---

## Author

**Elif Erkişi**  
Computer Engineering  
Industrial Engineering Minor  
MEF University
