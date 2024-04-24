import pandas as pd
import numpy as np
from scipy.stats import norm
import plotly.graph_objects as go
import streamlit as st

def load_data():
    path_to_csvs = './'
    months = [
        'Load DetailApril2023',
        'Load DetailAugust2023.1-15',
        'Load DetailSeptember2023',
        'Load DetailAugust2023.16-31',
        'Load DetailDecember1-15.2023',
        'Load DetailDecember16-31.2023',
        'Load DetailFebruary2023',
        'Load DetailJanuary2023',
        'Load DetailJuly2023',
        'Load DetailJUNE2023',
        'Load DetailMarch2023',
        'Load DetailMay2023',
        'Load DetailNovember1-15.2023',
        'Load DetailNovember16-30.2023'
    ]  

    data = []

    for month in months:
        file_path = f'{path_to_csvs}{month}.csv'
        try:
            month_data = pd.read_csv(file_path)
            if 'Tonnage' in month_data.columns and 'Truck Factor' in month_data.columns and 'Shovel' in month_data.columns:
                month_data = month_data[month_data['Truck Factor'] > 0]
                data.append(month_data)
          
        except FileNotFoundError:
            print(f"CSV file not found for {month}. Skipping...")

    return data

def load_shovel_fill_data(data, shovel):
    shovel_fill_data = []

    for df in data:
        df = df[df['Shovel'].isin(shovel)]
        truck_fill_percentage = (df['Tonnage'] / df['Truck Factor']) * 100
        shovel_fill_data.extend(truck_fill_percentage.dropna())

    return shovel_fill_data

def plot_distribution(shovel_fill_data, shovel, desired_mean=100, desired_std=5):
    if not shovel_fill_data:
        st.write("No data available for the selected shovel(s).")
        return

    actual_mean = np.mean(shovel_fill_data)
    actual_std = np.std(shovel_fill_data)

    x_min = 65
    x_max = 125
    x_range = np.linspace(x_min, x_max, 200)

    actual_distribution_y = norm.pdf(x_range, actual_mean, actual_std)
    desired_distribution_y = norm.pdf(x_range, desired_mean, desired_std)

    mean_std_text = (f"<b>Actual Mean:</b> {actual_mean:.2f}%<br>"
                     f"<b>Actual Std Dev:</b> {actual_std:.2f}%<br>"
                     f"<b>Desired Mean:</b> {desired_mean}%<br>"
                     f"<b>Desired Std Dev:</b> {desired_std}%")

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=x_range, y=actual_distribution_y, mode='lines', name='Actual Distribution', line=dict(color='red')))
    fig.add_trace(go.Scatter(x=x_range, y=desired_distribution_y, mode='lines', name='Desired Distribution', line=dict(color='#00B7F1')))

    fig.add_trace(go.Scatter(x=[actual_mean, actual_mean], y=[0, max(actual_distribution_y)], mode='lines', name='Actual Mean', line=dict(color='red', dash='dash')))
    fig.add_trace(go.Scatter(x=[desired_mean, desired_mean], y=[0, max(desired_distribution_y)], mode='lines', name='Desired Mean', line=dict(color='#00B7F1', dash='dash')))

    fig.add_annotation(
        text=mean_std_text,
        align='left',
        showarrow=False,
        xref='paper',
        yref='paper',
        x=1,
        y=1,
        bordercolor='black',
        borderwidth=1,
        bgcolor='white',
        xanchor='left',
        yanchor='top'
    )

    fig.update_layout(
        title=f'Actual vs Desired Truck Fill Distribution for {shovel}',
        xaxis_title='Truck Fill %',
        yaxis_title='Probability Density',
        legend_title='Legend',
        height=500,  # Adjusted height to fit the screen
        width=900,   # Adjusted width to fit the screen
        legend=dict(
            x=1.05,
            y=0.5,
            xanchor='left',
            yanchor='top',
            bgcolor='rgba(255,255,255,0.5)'
        ),
        xaxis=dict(range=[x_min, x_max])
    )

    return fig

def process_loaded_data(data):
    all_data = pd.concat([df for df in data])
    all_data['Time Full'] = pd.to_datetime(all_data['Time Full'], errors="coerce").dropna()
    all_data['Hour'] = all_data['Time Full'].dt.hour
    all_data['Shift'] = all_data['Hour'].apply(lambda x: 'Day' if 7 <= x < 19 else 'Night')
    all_data['Truck fill (%)'] = (all_data['Tonnage'] / all_data['Truck Factor']) * 100
    all_data['Month'] = all_data['Time Full'].dt.month
    all_data['Season'] = all_data['Month'].apply(month_to_season)
    all_data['Year'] = all_data['Time Full'].dt.year
    all_data = all_data.dropna(subset=['Time Full'])
    return all_data

def month_to_season(month):
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    elif month in [9, 10, 11]:
        return 'Fall'

def create_timeseries_plot(data):
    data = adjust_hours_for_plot(data)
    average_fill_by_hour_shift = data.groupby(['Adjusted Hour', 'Shift'])['Truck fill (%)'].mean().reset_index()

    trace_day = go.Scatter(
        x=average_fill_by_hour_shift[average_fill_by_hour_shift['Shift'] == 'Day']['Adjusted Hour'],
        y=average_fill_by_hour_shift[average_fill_by_hour_shift['Shift'] == 'Day']['Truck fill (%)'],
        mode='lines',
        name='Day Shift',
        line=dict(color='red')
    )

    trace_night = go.Scatter(
        x=average_fill_by_hour_shift[average_fill_by_hour_shift['Shift'] == 'Night']['Adjusted Hour'],
        y=average_fill_by_hour_shift[average_fill_by_hour_shift['Shift'] == 'Night']['Truck fill (%)'],
        mode='lines',
        name='Night Shift',
        line=dict(color='blue')
    )

    layout = go.Layout(
        title='Average Truck Fill by Hour and Shift (7 AM to 7 AM)',
        xaxis=dict(title='Hour (7 AM to 7 AM)', dtick=1, tickvals=list(range(24)), ticktext=[f"{(h+7)%24}:00" for h in range(24)]),
        yaxis=dict(title='Average Truck Fill (%)')
    )

    fig = go.Figure(data=[trace_day, trace_night], layout=layout)

    return fig

def adjust_hours_for_plot(df):
    df['Adjusted Hour'] = df['Hour'].apply(lambda x: (x + 17) % 24)
    df.sort_values(by=['Adjusted Hour'], inplace=True)
    return df

def calculate_material_increase(current_mean, current_std, desired_mean=100, desired_std=5):
    z_current = (desired_mean - current_mean) / current_std
    z_desired = 0

    potential_increase = norm.cdf(z_current) - norm.cdf(z_desired)

    return potential_increase * 100

def load_truck_fill_data(data, shovels):
    data = data[data['Shovel'].isin(shovels)]
    all_months_data = []
    total_improvement = 0

    month_aggregated_data = {}

    data['Month'] = data['Time Full'].dt.strftime('%B')

    data['Month-Year'] = data['Month'].astype(str) + ' ' + data['Year'].astype(str)
    data['Year'] = data['Year'].astype(int)
    data = data.sort_values(by='Year')
    for month in data['Month-Year'].unique():
        month_data = data[data['Month-Year'] == month]

        current_truck_fill = (month_data['Tonnage'] / month_data['Truck Factor']) * 100
        current_material_moved = month_data['Tonnage'].sum()
        desired_material_moved = current_material_moved * (1 + calculate_material_increase(current_truck_fill.mean(), 100, 100, 5) / 100)
        improvement = desired_material_moved - current_material_moved
        total_improvement += improvement

        month_year_key = (month_data['Month'].iloc[0], month_data['Year'].iloc[0])

        if month_year_key in month_aggregated_data:
            month_aggregated_data[month_year_key]['Current Material'] += current_material_moved
            month_aggregated_data[month_year_key]['Desired Material'] += desired_material_moved
            month_aggregated_data[month_year_key]['Improvement'] += improvement
        else:
            month_aggregated_data[month_year_key] = {
                'Month': month_data['Month'].iloc[0],
                'Year': month_data['Year'].iloc[0],
                'Current Truck Fill Rate': f"{current_truck_fill.mean():.2f}%",
                'Desired Truck Fill Rate': "100%",
                'Current Material': current_material_moved,
                'Desired Material': desired_material_moved,
                'Improvement': improvement
            }

    all_months_data = list(month_aggregated_data.values())

    total_row = {
        'Month': 'Total',
        'Year': '',
        'Current Truck Fill Rate': '',
        'Desired Truck Fill Rate': '',
        'Current Material': f"{sum(month_data['Current Material'] for month_data in all_months_data):.2e}",
        'Desired Material': f"{sum(month_data['Desired Material'] for month_data in all_months_data):.2e}",
        'Improvement': f"{improvement:+.2e}"  # Using '+.2e' to ensure the sign is included in the scientific notation

    }
    result_df = pd.DataFrame(all_months_data)

    result_df['Current Material'] = result_df['Current Material'].apply(lambda x: f'{float(x):.2e}')
    result_df['Desired Material'] = result_df['Desired Material'].apply(lambda x: f'{float(x):.2e}')
    result_df['Improvement'] = result_df['Improvement'].apply(lambda x: f'{float(x):.2e}')
    result_df = pd.concat([result_df, pd.DataFrame([total_row])], ignore_index=True)

    return result_df

def generate_markdown_explanation(actual_mean, actual_std, desired_mean, desired_std, shovel):
    explanation = f"""
    The purpose of this analysis is to evaluate the potential improvements in operational efficiency with the implementation of ShovelMetrics™ Payload Monitoring (SM-PLM). By analyzing the truck fill distribution data, we aim to identify areas where optimizations can be made to enhance productivity and reduce operational risks. To illustrate potential improvements with SM-PLM for shovel '{shovel}', the below distributions are shown with a target fill of {desired_mean}% and a standard deviation of {desired_std}% to emulate the distribution with SM-PLM.
    """
    return explanation


import pandas as pd
import numpy as np
from scipy.stats import norm
import plotly.graph_objects as go
import streamlit as st

# Define your functions here...

def main():
    st.title("Potential Improvements to Operational Efficiency with ShovelMetrics™ PLM")
    st.markdown("Prepared for: Aktogay Mine")
    st.markdown("Date: 2024-04-22")
    intro_placeholder = st.empty()

    data = load_data()

    all_shovels = list(set([value for df in data for value in df['Shovel'].unique() if 'Shovel' in df.columns]))
    all_shovels.append('All')

    selected_shovels = st.sidebar.multiselect("Select Shovel", all_shovels, default=['All'])

    if 'All' in selected_shovels or len(selected_shovels) == 0:
        selected_shovels = [shovel for shovel in all_shovels if shovel != 'All']

    if len(selected_shovels)==0:
        selected_shovel = ['All']

    if 'All' in selected_shovels:
        selected_shovels = all_shovels
        selected_shovels.remove('All')

    selected_mean = st.sidebar.slider("Select Mean (%)", 98, 110, 100, step=1)
    selected_std = st.sidebar.slider("Select Standard Deviation (%)", 1, 10, 5, step=1)

    explanation_placeholder = st.empty()

    shovel_fill_data = load_shovel_fill_data(data, selected_shovels)
    fig = plot_distribution(shovel_fill_data, selected_shovels, selected_mean, selected_std)
    st.plotly_chart(fig, use_container_width=True)

    actual_mean = np.mean(shovel_fill_data) if shovel_fill_data else 0
    actual_std = np.std(shovel_fill_data) if shovel_fill_data else 0
    explanation_text = generate_markdown_explanation(actual_mean, actual_std, selected_mean, selected_std, selected_shovels)
    explanation_placeholder.markdown(explanation_text)

    data = process_loaded_data(data)
    data = data[data['Shovel'].isin(selected_shovels)]
    fig = create_timeseries_plot(data)
    st.plotly_chart(fig, use_container_width=True)

    day_mean = data[data['Shift'] == 'Day']['Truck fill (%)'].mean()
    night_mean = data[data['Shift'] == 'Night']['Truck fill (%)'].mean()
    peak_day = data[(data['Shift'] == 'Day') & (data['Hour'].between(10, 16))]['Truck fill (%)'].mean()
    peak_night = data[(data['Shift'] == 'Night') & ((data['Hour'] >= 22) | (data['Hour'] <= 4))]['Truck fill (%)'].mean()

    if day_mean > night_mean:
        shift_message = ("During day shifts, the average truck fill percentage is {:.2f}%, "
                     " the night shifts' average is {:.2f}%. ")
    else:
        shift_message = ("During night shifts, the average truck fill percentage is {:.2f}%, "
                     " the day shifts' average of {:.2f}%. ")

    st.write(shift_message.format(day_mean, night_mean))

    peak_message = ("The peak average truck fill percentage during day shifts occurs between 10 AM and 4 PM with {:.2f}%, "
                "while during night shifts, it occurs between 10 PM and 4 AM with {:.2f}%.")

    st.write(peak_message.format(peak_day, peak_night))

    selected_title = f"Material Destination Distribution for {selected_shovels}"
    st.markdown(f'<h1 style="font-size: 24px;">{selected_title}</h1>', unsafe_allow_html=True)
    st.markdown("The chart below illustrates the proportion of trucks sent to one of four locations: Crusher, Stockpiles, DLP, and Other. Kindly note that We are only analyzing truck fill for trucks that are sent to the crusher.")
    shovel_data = data[data['Shovel'].isin(selected_shovels)]
    data_cleaned = shovel_data.dropna()

    def categorize_destination(destination):
        if destination.upper().startswith('CRUSHER'):
            return 'Crusher'
        elif destination.startswith('STK'):
            return 'STK'
        elif destination.startswith('DLP'):
            return 'DLP'
        else:
            return 'Other'

    data_cleaned['Destination Category'] = data_cleaned['Assigned Dump'].apply(categorize_destination)

    destination_counts = data_cleaned['Destination Category'].value_counts()

    other_destinations = data_cleaned[data_cleaned['Destination Category'] == 'Other']['Assigned Dump'].unique()

    st.markdown("The 'Other' category comprises the following destinations:")
    st.write(", ".join(other_destinations))

    st.markdown(f'<h2 style="font-size: 20px;">Material Destination Distribution</h2>', unsafe_allow_html=True)
    fig_destination = go.Figure(data=[go.Pie(labels=destination_counts.index, values=destination_counts.values, hole=0.3)])
    fig_destination.update_traces(textinfo='percent+label', textposition='inside')
    st.plotly_chart(fig_destination)

    material_data = data_cleaned['Material'].value_counts()
    fig_material_grade = go.Figure(data=[go.Pie(labels=material_data.index, values=material_data.values, hole=0.3)])
    fig_material_grade.update_traces(textinfo='percent+label', textposition='inside')
    st.markdown(f'<h2 style="font-size: 20px;">Material Grade Distribution</h2>', unsafe_allow_html=True)
    st.plotly_chart(fig_material_grade)

    shovel_data = data[(data['Shovel'].isin(selected_shovels)) & (data['Tonnage'] != 0) & (data['Truck Factor'] != 0)].dropna(subset=['Truck Factor', 'Tonnage'])
    shovel_data['Truck Fill Rate (%)'] = (shovel_data['Tonnage'] / shovel_data['Truck Factor']) * 100

    monthly_performance = shovel_data.groupby(['Season', 'Month'])['Truck Fill Rate (%)'].mean().reset_index()

    colors = {'Winter': '#1f77b4', 'Spring': '#ff7f0e', 'Summer': '#2ca02c', 'Fall': '#d62728'}

    monthly_performance['Color'] = monthly_performance['Season'].map(colors)

    month_names = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun', 7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}

    fig_monthly = go.Figure()
    for season, color in colors.items():
        season_data = monthly_performance[monthly_performance['Season'] == season]
        fig_monthly.add_trace(go.Bar(x=season_data['Month'].map(month_names), y=season_data['Truck Fill Rate (%)'],
                                    name=f'{season} Average', marker_color=color))
        fig_monthly.add_trace(go.Scatter(x=season_data['Month'].map(month_names), y=[season_data['Truck Fill Rate (%)'].mean()] * len(season_data),
                                        mode='lines+markers', name=f'{season} Average', line=dict(color=color, dash='dash', width=2),
                                        marker=dict(color=color, size=8)))

    fig_monthly.update_layout(xaxis_title='Month', yaxis_title='Average Truck Fill Rate (%)',
                            template='plotly_white', yaxis=dict(range=[80, 105]),
                            title=dict(text=f'Monthly Truck Fill Rate Trends for {selected_shovels}', font=dict(size=18, color='black', family="Arial")))
    st.plotly_chart(fig_monthly, use_container_width=True)

       # ---------------------------- Tabular View ---------------------------------------------------
    # Load your truck fill rate data
    results_df = load_truck_fill_data(data, selected_shovels)

    # Sort the DataFrame by year and month
    results_df['Month'] = pd.Categorical(results_df['Month'], categories=[
        'January', 'February', 'March', 'April', 'May', 'June',
        'July', 'August', 'September', 'October', 'November', 'December', 'Total'
    ], ordered=True)
    results_df = results_df.sort_values(by=['Year', 'Month'])
    results_df = results_df.reset_index(drop=True)

    shovel_text = ", ".join(selected_shovels) if selected_shovels else "All Shovels"
    st.markdown(f"<h4><b>Current and Desired Truck Fill Rates for {shovel_text}</b></h4>", unsafe_allow_html=True)

    # Add CSS styling to the header of the table to change the background color
    header_html = """
    <style>
    th {
    background-color: #00B7F1; 
    }

    th div {
    color: white;
    }
    </style>
    """
    st.markdown(header_html, unsafe_allow_html=True)
    st.table(results_df)
    
if __name__ == '__main__':
    main()
