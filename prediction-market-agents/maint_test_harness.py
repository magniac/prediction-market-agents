import os
import random
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------
# Base Agent: Global cash and per-event inventory tracking
# ----------------------------
class BaseAgent:
    def __init__(self, initial_cash=50000, outcomes=None):
        # Global cash is carried across events.
        self.cash = initial_cash
        # Outcomes—for prediction markets these might be "prob_yes" and "prob_no"
        self.outcomes = outcomes if outcomes is not None else ["prob_yes", "prob_no"]
        # Holdings will be stored by event. Each key is an event and its value is a dictionary of outcome positions.
        self.holdings = {}
        self.trade_history = []      # Each trade record includes event information.
        self.portfolio_history = []  # List of (timestamp, portfolio_value) tuples.

    def update_portfolio_history(self, timestamp, last_prices):
        # Compute portfolio value as global cash plus the value of all positions.
        portfolio_value = self.cash
        for event, prices in last_prices.items():
            if event in self.holdings:
                for outcome in self.outcomes:
                    portfolio_value += self.holdings[event][outcome] * prices[outcome]
        self.portfolio_history.append((timestamp, portfolio_value))


# ----------------------------
# Random Agent: Operates row-by-row on the global timeline
# ----------------------------
class RandomAgent(BaseAgent):
    def act(self, timestamp, event, price_data):
        # Ensure there is an inventory record for this event.
        if event not in self.holdings:
            self.holdings[event] = {o: 0 for o in self.outcomes}
        # Randomly choose an outcome and an action.
        outcome = random.choice(list(price_data.keys()))
        action = random.choice(["buy", "sell", "hold"])
        price = price_data[outcome]
        if action == "buy" and self.cash >= price:
            self.cash -= price
            self.holdings[event][outcome] += random.choice([1,2,3,4,5])
            self.trade_history.append({
                "timestamp": timestamp, 
                "event": event,
                "outcome": outcome, 
                "action": "buy", 
                "price": price
            })
        elif action == "sell" and self.holdings[event][outcome] > 0:
            self.cash += price
            self.holdings[event][outcome] -= min(self.holdings[event][outcome], random.choice([1,2,3,4,5]))
            self.trade_history.append({
                "timestamp": timestamp, 
                "event": event,
                "outcome": outcome, 
                "action": "sell", 
                "price": price
            })
        else:
            self.trade_history.append({
                "timestamp": timestamp, 
                "event": event,
                "outcome": outcome, 
                "action": "hold", 
                "price": price
            })
    
    def run(self, data):
        """
        Process the data sorted globally by timestamp.
        For each row, update the price for that event, decide an action,
        and update the global portfolio value.
        """
        # Dictionary to hold the most recent price data for each event.
        last_prices = {}
        # Iterate through the rows in time order.
        for _, row in data.iterrows():
            timestamp = row["timestamp"]
            event = row["event"]
            # Build the price_data dictionary for this row.
            price_data = {o: float(row[o]) for o in self.outcomes}
            # Update the last observed price for this event.
            last_prices[event] = price_data
            # Call the agent's act function, which updates inventory for the event.
            self.act(timestamp, event, price_data)
            # Compute the global portfolio value and record it.
            self.update_portfolio_history(timestamp, last_prices)



# ----------------------------
# Mean Reversion Agent
# ----------------------------
class MeanReversionAgent(BaseAgent):
    def __init__(self, initial_cash=50000, threshold=0.05, alpha=0.1, outcomes=None):
        """
        threshold: fraction above/below the running average that triggers an action.
        alpha: smoothing factor for the exponential moving average.
        """
        super().__init__(initial_cash, outcomes)
        self.threshold = threshold
        self.alpha = alpha
        # Running averages stored per event and outcome.
        self.running_avg = {}  # structure: { event: { outcome: running_avg, ... }, ... }
        
    def act(self, timestamp, event, price_data):
        # Ensure holdings and running average exist for this event.
        if event not in self.holdings:
            self.holdings[event] = {o: 0 for o in self.outcomes}
        if event not in self.running_avg:
            # Initialize running average with current price for each outcome.
            self.running_avg[event] = {o: price_data[o] for o in self.outcomes}
        
        # For each outcome, update the running average and decide on trade.
        for outcome in self.outcomes:
            current_price = price_data[outcome]
            # Update exponential moving average.
            prev_avg = self.running_avg[event][outcome]
            new_avg = self.alpha * current_price + (1 - self.alpha) * prev_avg
            self.running_avg[event][outcome] = new_avg

            # Calculate deviation relative to the running average.
            deviation = (current_price - new_avg) / new_avg if new_avg != 0 else 0

            # Mean reversion logic:
            if deviation > self.threshold:
                # Price is above average; sell if holding.
                if self.holdings[event][outcome] > 0:
                    self.cash += current_price
                    self.holdings[event][outcome] -= min(self.holdings[event][outcome], random.choice([1,2,3,4,5]))
                    self.trade_history.append({
                        "timestamp": timestamp, "event": event, "outcome": outcome,
                        "action": "sell", "price": current_price, "deviation": deviation
                    })
                else:
                    self.trade_history.append({
                        "timestamp": timestamp, "event": event, "outcome": outcome,
                        "action": "hold", "price": current_price, "deviation": deviation
                    })
            elif deviation < -self.threshold:
                # Price is below average; buy if cash is available.
                if self.cash >= current_price:
                    self.cash -= current_price
                    self.holdings[event][outcome] += random.choice([1,2,3,4,5])
                    self.trade_history.append({
                        "timestamp": timestamp, "event": event, "outcome": outcome,
                        "action": "buy", "price": current_price, "deviation": deviation
                    })
                else:
                    self.trade_history.append({
                        "timestamp": timestamp, "event": event, "outcome": outcome,
                        "action": "hold", "price": current_price, "deviation": deviation
                    })
            else:
                # No significant deviation; do nothing.
                self.trade_history.append({
                    "timestamp": timestamp, "event": event, "outcome": outcome,
                    "action": "hold", "price": current_price, "deviation": deviation
                })
    
    def run(self, data):
        """
        Iterate over rows in the global timeline.
        For each row, update running averages and call act().
        """
        last_prices = {}
        for _, row in data.iterrows():
            timestamp = row["timestamp"]
            event = row["event"]
            price_data = {o: float(row[o]) for o in self.outcomes}
            # Update last observed price for this event.
            last_prices[event] = price_data
            self.act(timestamp, event, price_data)
            self.update_portfolio_history(timestamp, last_prices)


# ----------------------------
# Momentum Agent
# ----------------------------
class MomentumAgent(BaseAgent):
    def __init__(self, initial_cash=50000, momentum_threshold=0.001, outcomes=None):
        """
        momentum_threshold: a threshold for the momentum value to trigger a trade.
        (Assumes the "momentum" column is available in the data.)
        """
        super().__init__(initial_cash, outcomes)
        self.momentum_threshold = momentum_threshold

    def act(self, timestamp, event, price_data, momentum):
        # Ensure an inventory record exists for this event.
        if event not in self.holdings:
            self.holdings[event] = {o: 0 for o in self.outcomes}
        
        # For demonstration, we'll use the momentum for "prob_yes" (or average momentum across outcomes)
        # to drive a decision on a randomly selected outcome.
        # You can refine this logic as needed.
        outcome = random.choice(list(price_data.keys()))
        current_price = price_data[outcome]
        # If momentum is strongly positive, we expect price to continue rising; buy.
        if momentum > self.momentum_threshold:
            if self.cash >= current_price:
                self.cash -= current_price
                self.holdings[event][outcome] += random.choice([1,2,3,4,5])
                self.trade_history.append({
                    "timestamp": timestamp, "event": event, "outcome": outcome,
                    "action": "buy", "price": current_price, "momentum": momentum
                })
            else:
                self.trade_history.append({
                    "timestamp": timestamp, "event": event, "outcome": outcome,
                    "action": "hold", "price": current_price, "momentum": momentum
                })
        # If momentum is strongly negative, we expect a downturn; sell if holding.
        elif momentum < -self.momentum_threshold:
            if self.holdings[event][outcome] > 0:
                self.cash += current_price
                self.holdings[event][outcome] -= min(self.holdings[event][outcome], random.choice([1,2,3,4,5]))
                self.trade_history.append({
                    "timestamp": timestamp, "event": event, "outcome": outcome,
                    "action": "sell", "price": current_price, "momentum": momentum
                })
            else:
                self.trade_history.append({
                    "timestamp": timestamp, "event": event, "outcome": outcome,
                    "action": "hold", "price": current_price, "momentum": momentum
                })
        else:
            self.trade_history.append({
                "timestamp": timestamp, "event": event, "outcome": outcome,
                "action": "hold", "price": current_price, "momentum": momentum
            })
    
    def run(self, data):
        """
        Process data on the global timeline.
        For each row, use the momentum column to drive a decision.
        """
        last_prices = {}
        for _, row in data.iterrows():
            timestamp = row["timestamp"]
            event = row["event"]
            price_data = {o: float(row[o]) for o in self.outcomes}
            # Retrieve the momentum value (assumed to be provided in the data).
            momentum = float(row.get("momentum", 0))
            # Update last observed price for this event.
            last_prices[event] = price_data
            self.act(timestamp, event, price_data, momentum)
            self.update_portfolio_history(timestamp, last_prices)


# ----------------------------
# Test Harness: Run Agents on Global Timeline and Plot Results
# ----------------------------
def run_agents(agents, data):
    """
    Run each agent's simulation on the globally sorted data.
    Each agent must implement run(data), which populates its portfolio_history.
    Returns a dictionary mapping agent names to their portfolio history.
    """
    results = {}
    for agent in agents:
        # Reset history if running multiple tests.
        agent.trade_history = []
        agent.portfolio_history = []
        # Run the simulation on a copy of the data.
        agent.run(data.copy())
        results[agent.__class__.__name__] = agent.portfolio_history
    return results

def plot_results(results):
    """
    Plot portfolio value over time for each agent.
    Each portfolio history is a list of (timestamp, portfolio_value) tuples.
    """
    plt.figure(figsize=(12, 6))
    for agent_name, history in results.items():
        df = pd.DataFrame(history, columns=["timestamp", "portfolio_value"])
        df.sort_values("timestamp", inplace=True)
        plt.plot(df["timestamp"], df["portfolio_value"], label=agent_name)
        plt.show()
    plt.xlabel("Time")
    plt.ylabel("Portfolio Value ($)")
    plt.title("Agent Performance Over Time")
    plt.legend()
    plt.show()


# ----------------------------
# Contrarian Agent
# ----------------------------
class ContrarianAgent(BaseAgent):
    def __init__(self, initial_cash=50000, contrarian_threshold=0.1, outcomes=None):
        """
        contrarian_threshold: if the absolute spread is greater than this value, trigger a contrarian action.
        For example, if spread > threshold then the market is very bullish (high prob_yes), so bet on the opposite.
        """
        super().__init__(initial_cash, outcomes)
        self.contrarian_threshold = contrarian_threshold

    def act(self, timestamp, event, price_data, spread):
        # Ensure that we have an inventory record for the event.
        if event not in self.holdings:
            self.holdings[event] = {o: 0 for o in self.outcomes}

        # Contrarian logic based on spread:
        if spread > self.contrarian_threshold:
            # Market is bullish (prob_yes is high).  
            # If holding prob_yes, sell it to take profits. Otherwise, try to buy prob_no.
            if self.holdings[event]["prob_yes"] > 0:
                price = price_data["prob_yes"]
                self.cash += price
                self.holdings[event]["prob_yes"] -= min(self.holdings[event]["prob_yes"], random.choice([1,2,3,4,5]))
                self.trade_history.append({
                    "timestamp": timestamp, 
                    "event": event, 
                    "outcome": "prob_yes",
                    "action": "sell", 
                    "price": price, 
                    "spread": spread,
                    "reason": "contrarian: bullish market, selling prob_yes"
                })
            elif self.cash >= price_data["prob_no"]:
                price = price_data["prob_no"]
                self.cash -= price
                self.holdings[event]["prob_no"] += random.choice([1,2,3,4,5])
                self.trade_history.append({
                    "timestamp": timestamp, 
                    "event": event, 
                    "outcome": "prob_no",
                    "action": "buy", 
                    "price": price, 
                    "spread": spread,
                    "reason": "contrarian: bullish market, buying prob_no"
                })
            else:
                self.trade_history.append({
                    "timestamp": timestamp, 
                    "event": event, 
                    "action": "hold", 
                    "price": None,
                    "spread": spread,
                    "reason": "contrarian: bullish market, no action"
                })
        elif spread < -self.contrarian_threshold:
            # Market is bearish (prob_no is high).
            # If holding prob_no, sell it. Otherwise, try to buy prob_yes.
            if self.holdings[event]["prob_no"] > 0:
                price = price_data["prob_no"]
                self.cash += price
                self.holdings[event]["prob_no"] -= min(self.holdings[event]["prob_no"], random.choice([1,2,3,4,5]))
                self.trade_history.append({
                    "timestamp": timestamp, 
                    "event": event, 
                    "outcome": "prob_no",
                    "action": "sell", 
                    "price": price, 
                    "spread": spread,
                    "reason": "contrarian: bearish market, selling prob_no"
                })
            elif self.cash >= price_data["prob_yes"]:
                price = price_data["prob_yes"]
                self.cash -= price
                self.holdings[event]["prob_yes"] += random.choice([1,2,3,4,5])
                self.trade_history.append({
                    "timestamp": timestamp, 
                    "event": event, 
                    "outcome": "prob_yes",
                    "action": "buy", 
                    "price": price, 
                    "spread": spread,
                    "reason": "contrarian: bearish market, buying prob_yes"
                })
            else:
                self.trade_history.append({
                    "timestamp": timestamp, 
                    "event": event, 
                    "action": "hold", 
                    "price": None,
                    "spread": spread,
                    "reason": "contrarian: bearish market, no action"
                })
        else:
            # If spread is not extreme, hold.
            self.trade_history.append({
                "timestamp": timestamp, 
                "event": event, 
                "action": "hold", 
                "price": None,
                "spread": spread,
                "reason": "contrarian: spread within threshold"
            })
    
    def run(self, data):
        """
        Iterate over rows in the global timeline.
        For each row, extract the price data and the spread, then call act().
        """
        last_prices = {}
        for _, row in data.iterrows():
            timestamp = row["timestamp"]
            event = row["event"]
            price_data = {o: float(row[o]) for o in self.outcomes}
            # Read the spread from the data.
            spread = float(row.get("spread", 0))
            # Update the last observed price for this event.
            last_prices[event] = price_data
            # Call the contrarian logic.
            self.act(timestamp, event, price_data, spread)
            # Update the global portfolio history.
            self.update_portfolio_history(timestamp, last_prices)


def split_events_by_end_time(df, train_ratio=0.7):
    """
    Splits the dataset into train and test by entire events, based on each event's end_time.
    1) For each event, find min(timestamp) and max(timestamp).
    2) Sort events by their max(timestamp).
    3) Take the first train_ratio% of events as the train set, the rest as test.
    """
    # Group by event to find start and end times
    summary_df = (
        df.groupby("event")["timestamp"]
        .agg(["min", "max"])
        .rename(columns={"min": "start_time", "max": "end_time"})
        .reset_index()
    )
    # Sort events by end_time
    summary_df = summary_df.sort_values("end_time")

    # Decide how many events go to train vs test
    n_events = len(summary_df)
    split_idx = int(n_events * train_ratio)
    
    # Get the list of events for train vs test
    train_events = summary_df.iloc[:split_idx]["event"].tolist()
    test_events = summary_df.iloc[split_idx:]["event"].tolist()

    # Build the train and test DataFrames
    train_df = df[df["event"].isin(train_events)].copy()
    test_df = df[df["event"].isin(test_events)].copy()

    # Sort each subset by timestamp (optional but recommended)
    train_df.sort_values("timestamp", inplace=True)
    test_df.sort_values("timestamp", inplace=True)
    
    return train_df, test_df

# ----------------------------
# Main Execution Example
# ----------------------------
if __name__ == "__main__":
    random.seed(1337)
    # Load your processed category dataset.
    # The CSV should have columns: timestamp, prob_yes, prob_no, spread, momentum, elapsed_fraction, event.
    data = pd.read_csv("processed_culture_category.csv")
    # Ensure the timestamp column is parsed as datetime.
    data["timestamp"] = pd.to_datetime(data["timestamp"])
    # Sort the data globally by timestamp.
    print(data.head())
    train_ratio = 0.7
    train_df, test_df = split_events_by_end_time(data, train_ratio=train_ratio)
    print(f"Train events: {train_df['event'].nunique()}, Test events: {test_df['event'].nunique()}")

    #data = data.sort_values(by="timestamp")
    cash_start = 50000
    # Instantiate agents.
    random_agent = RandomAgent(initial_cash=cash_start, outcomes=["prob_yes", "prob_no"])
    mean_rev_agent = MeanReversionAgent(initial_cash=cash_start, threshold=0.05, alpha=0.1, outcomes=["prob_yes", "prob_no"])
    momentum_agent = MomentumAgent(initial_cash=cash_start, momentum_threshold=0.001, outcomes=["prob_yes", "prob_no"])
    contrarian_agent = ContrarianAgent(initial_cash=cash_start, contrarian_threshold=0.1, outcomes=["prob_yes", "prob_no"])

    agents = [random_agent, mean_rev_agent, momentum_agent, contrarian_agent]

    # 1) Run on train events (rule-based won't "train" but we keep the pipeline consistent)
    train_results = run_agents(agents, train_df)
    plot_results(train_results)#, title="Train Results (Event-Based Split)")

    # 2) Reset the agents if you want them to start test fresh:
    for agent in agents:
        agent.cash = cash_start
        agent.holdings = {}
        agent.trade_history = []
        agent.portfolio_history = []
        # For learning agents (like a DQN), you'd keep the learned policy but reset the portfolio.

    # 3) Run on test events
    test_results = run_agents(agents, test_df)
    plot_results(test_results)#, title="Test Results (Event-Based Split)")
