from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib.ticker as mtick
import numpy as np



netflix_stocks = pd.read_csv('NFLX.csv')
print(netflix_stocks)



dowjones_stocks = pd.read_csv('DJI.csv')
print(dowjones_stocks)



netflix_stocks_quarterly = pd.read_csv('NFLX_daily_by_quarter.csv')
print(netflix_stocks_quarterly)



netflix_stocks.head()



netflix_stocks.rename(columns={'Adj Close': 'Price'}, inplace=True)



netflix_stocks.head()



dowjones_stocks.rename(columns={'Adj Close': 'Price'}, inplace=True)
netflix_stocks_quarterly.rename(columns={'Adj Close': 'Price'}, inplace=True)
print(dowjones_stocks.head())
print(netflix_stocks_quarterly.head())



fig = plt.figure(figsize=(8,8))
sns.set_style('darkgrid', {'axes.linewidth': 2, 'axes.edgecolor':'black'})
sns.set_context("talk")
ax = sns.violinplot(data=netflix_stocks_quarterly, x='Quarter', y='Price')
ax.set_title("Distribution of 2017 Netflix Stock Prices by Quarter")
ax.set_xlabel('Closing Stock Price')
ax.set_ylabel('Business Quarters in 2017')
fmt = '${x:,.0f}'
dollar_tick = mtick.StrMethodFormatter(fmt)
ax.yaxis.set_major_formatter(dollar_tick)
#plt.savefig('Stock-Price-Distribution.png', bbox_inches='tight')
plt.show()



min_price = min(netflix_stocks_quarterly['Price'])
max_price = max(netflix_stocks_quarterly['Price'])
print(f'The lowest price was:\t{min_price}')
print(f'The highest price was:\t{max_price}')



fig = plt.figure(figsize=(8,8))
x_positions = [1, 2, 3, 4]
chart_labels = ["1Q2017","2Q2017","3Q2017","4Q2017"]
earnings_actual =[.4, .15,.29,.41]
earnings_estimate = [.37,.15,.32,.41 ]
plt.scatter(x_positions, earnings_actual, c='red', alpha=0.5, s=45)
plt.scatter(x_positions, earnings_estimate, c='blue', alpha=0.5, s=45)
plt.xticks(x_positions, chart_labels)
plt.title('Yahoo earnings estimate compared to actual earnings')
plt.xlabel('Quarter (2017)')
plt.ylabel('Earnings')
# plt.savefig('estimates-actual-earnings.png', bbox_inches='tight')
plt.show()



sns.set('talk')
sns.set_style('darkgrid', {'axes.linewidth': 2, 'axes.edgecolor':'black'})
fig = plt.figure(figsize=(8,8))
# The metrics below are in billions of dollars
revenue_by_quarter = [2.79, 2.98,3.29,3.7]
earnings_by_quarter = [.0656,.12959,.18552,.29012]
quarter_labels = ["2Q2017","3Q2017","4Q2017", "1Q2018"]

# Revenue
n = 1  # This is our first dataset (out of 2)
t = 2 # Number of dataset
d = 4 # Number of sets of bars
w = 0.8 # Width of each bar
bars1_x = [t*element + w*n for element
             in range(d)]



# Earnings
n = 2  # This is our second dataset (out of 2)
t = 2 # Number of dataset
d = 4 # Number of sets of bars
w = 0.8 # Width of each bar
bars2_x = [t*element + w*n for element
             in range(d)]

middle_x = [ (a + b) / 2.0 for a, b in zip(bars1_x, bars2_x)]
labels = ["Revenue", "Earnings"]

plt.bar(bars1_x, revenue_by_quarter)
plt.bar(bars2_x, earnings_by_quarter)
plt.legend(labels)
plt.xticks(middle_x, quarter_labels)
plt.ylabel('Value')
plt.xlabel('Quarter')
plt.title('Revenue to Earnings comparison')
#lt.savefig('Revenue-Earnings-Comparison.png', bbox_inches='tight')
plt.show()



percentage_earnings = [earnings/revenue for revenue, earnings in zip(revenue_by_quarter,earnings_by_quarter)]
for i in range(4):
    print(f'Q{i+1}:\t\t\t{percentage_earnings[i]*100:.2f}%')
print()
print(f'Average percentage:\t{sum(percentage_earnings)/4*100:.2f}%')



dates = [date[-5:-3] for date in netflix_stocks['Date']]
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

sns.set('talk')
sns.set_style('ticks')
fig = plt.figure(figsize=(16,8))
# Left plot Netflix
ax1 = plt.subplot(2,2,1)
ax1.plot(netflix_stocks['Date'], netflix_stocks['Price'])
ax1.set_title('Stock Price')
ax1.set_xlabel('Date (2017)')
ax1.set_ylabel('Stock Price')
ax1.grid()
ax1.yaxis.set_major_formatter(dollar_tick)
plt.xticks(netflix_stocks['Date'], months)


# Right plot Dow Jones
ax2 = plt.subplot(2,2,2)
ax2.plot(dowjones_stocks['Date'], dowjones_stocks['Price'])
ax2.set_title('Dow Jones')
ax2.set_xlabel('Date (2017)')
ax2.set_ylabel('Stock Price')
ax2.grid()
ax2.yaxis.set_major_formatter(dollar_tick)
plt.xticks(dowjones_stocks['Date'], months)
plt.subplots_adjust(wspace=.4)

#plt.savefig('Stock-Price-Dow-Jones-Comp.png', bbox_inches='tight')
plt.show()



sns.set('talk')
sns.set_style('dark', {'axes.linewidth': 2, 'axes.edgecolor':'black'})
fig = plt.figure(figsize=(8,8))
ax3 = plt.subplot(1,1,1)
ax3.plot(netflix_stocks['Date'], netflix_stocks['Price'])
ax3.set_title('Stock Price Comparison')
ax3.set_xlabel('Date (2017)')
ax3.set_ylabel('Stock Price')
#ax3.grid()
ax3.yaxis.set_major_formatter(dollar_tick)
ax4 = ax3.twinx()
ax4.plot(dowjones_stocks['Date'], dowjones_stocks['Price'], c='orange')
ax4.yaxis.set_major_formatter(dollar_tick)
plt.xticks(netflix_stocks['Date'], months)

#plt.savefig('Stock-Price-Comparison.png', bbox_inches='tight')
plt.show()



# calculations
percentages = [(netflix/dow_jones)*100 for netflix, dow_jones in zip(netflix_stocks['Price'],dowjones_stocks['Price'])]
average_average = sum(percentages)/len(percentages)
x = [i for i in range(12)]
line_of_best_fit = np.poly1d(np.polyfit(x, percentages, 1))(np.unique(x))
dow_jones_line_of_best_fit = np.poly1d(np.polyfit(x, dowjones_stocks['Price'], 1))(np.unique(x))




# print averages
for i, month in enumerate(months):
    print(f'{month}:\t\t\t{percentages[i]:.2f}%')
print('-----------------------------')
print(f'Average percentage:\t{average_average:.2f}%')




# decide what to plot
lines = ['Percentage of Dow Jones Average per month', 'Percentage line of best fit', 'Total Percentage Average', 'Dow Jones Average stock prices', 'Dow Jones Average stock prices line of best fit']
to_plot = list()
for i, line in enumerate(lines):
    do_plot = input('Would you like to plot - '+line+' [y/n]:\n')
    if do_plot == 'y':
        to_plot.append(line)

sns.set('talk', rc={'legend.fontsize': 11})
sns.set_style('white')
        
# figure sorting
fig = plt.figure(figsize=(10,10))
ax1 = plt.subplot(1,1,1)

# plotting left scale
color='tab:blue'
if 'Percentage of Dow Jones Average per month' in to_plot :
    best_fit = 'tab:green'
else:
    best_fit = color
if 'Percentage of Dow Jones Average per month' in to_plot:
    ax1.plot(months, percentages, marker='', color=color)
if 'Percentage line of best fit' in to_plot:
    ax1.plot(np.unique(x), line_of_best_fit, c=best_fit)
if 'Total Percentage Average' in to_plot:
    plt.plot(months, [average_average for i in range(12)], c='black', alpha=0.2)


# making left scale nice
if 'Percentage of Dow Jones Average per month' in to_plot or 'Percentage line of best fit' in to_plot or 'Total Percentage Average' in to_plot:
    ax1.tick_params(axis='y', labelcolor=color)
    fmt = '{x:.2f}%'
    tick = mtick.StrMethodFormatter(fmt)
    ax1.yaxis.set_major_formatter(tick)
    ax1.set_ylim([0.68, 0.85])
    plt.ylabel('Percentage of Dow Jones Average', c=color)
    labels = [label for label in to_plot if label != 'Dow Jones Average stock prices']
    plt.legend(labels, loc=2)
    plt.grid(axis='y', linestyle='--', color=color)
    plt.title('NFLX Stock Price as a percentage of Dow Jones Average in 2017')
plt.xlabel('Month')

# right axis creation
if 'Dow Jones Average stock prices' in to_plot or 'Dow Jones Average stock prices line of best fit' in to_plot:
    if ['Dow Jones Average stock prices', 'Dow Jones Average stock prices line of best fit'] != to_plot and ['Dow Jones Average stock prices'] != to_plot and ['Dow Jones Average stock prices line of best fit'] != to_plot:
        ax2 = ax1.twinx()
    else:
        ax2 = plt.subplot(1,1,1)
        plt.title('Dow Jones Average Stock Prices')
    color = 'tab:red'
    ax2.set_ylabel('Dow Jones Average Stock Price', c=color)
    # Deciding what to plot
    if 'Dow Jones Average stock prices' in to_plot:
        ax2.plot(months, dowjones_stocks['Price'], color=color)
    if 'Dow Jones Average stock prices line of best fit' in to_plot:
        ax2.plot(months, dow_jones_line_of_best_fit, color='orange')
    # legend, grid, tick and ylim altering
    ax2.tick_params(axis='y', labelcolor=color)
    plt.grid(axis='y', linestyle=':', c=color, alpha=0.4)
    if 'Dow Jones Average stock prices' not in to_plot or 'Dow Jones Average stock prices line of best fit' not in to_plot:
        bbox_to_anchor_y = 0.99 - 0.03 * (len(to_plot)-1)
    else:
        bbox_to_anchor_y = 0.99 - 0.03 * (len(to_plot)-2)
    plt.legend(['Dow Jones Average stock prices', 'Dow Jones Average stock prices line of best fit'], loc=2, bbox_to_anchor=(0.001, bbox_to_anchor_y))
    ax2.set_ylim([19500, 25000])

#plt.savefig('Changeable-percentage-stock-price.png', bbox_inches='tight')
plt.show()
