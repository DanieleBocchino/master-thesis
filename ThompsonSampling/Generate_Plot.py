import os
import shutil
from matplotlib import pyplot as plt
import numpy as np
from pylab import rcParams
import seaborn as sns
from matplotlib.animation import FuncAnimation
import math
import imageio
from PIL import Image
from utils import load_config

config =load_config()
colors = plt.cm.tab20(np.linspace(0, 1, 20))


def generate_hist(y, list_element, arr, nchoices=6, select_element=0, name_plot='hist_plot', outputs_dir='outputs/train/'):

    # bandit chioce
    choice_egr, choice_agr2, choice_lts, choice_own_lts, choice_real = [
        list(nchoices*[0]) for i in range(5)]
    lst_choice_hist = [choice_egr, choice_agr2,
                       choice_lts, choice_own_lts, choice_real]

    for model in range(len(lst_choice_hist)):
        if (model == 4):
            for i in range(len(arr)-select_element):
                lst_choice_hist[model][arr[i]] += 1

        elif model == 3:
            for i in range(len(y)-select_element):
                for j in range(len(y[i])):
                    if y[i][j] == 1:
                        lst_choice_hist[model][j] += 1
        else:
            for i in range(len(list_element[model])-select_element):
                lst_choice_hist[model][list_element[model][i]] += 1

    # set width of bar
    barWidth = 0.15
    fig = plt.subplots(figsize=(12, 8))

    # Set position of bar on X axis
    br1 = np.arange(len(lst_choice_hist[0]))
    br2 = [x + barWidth for x in br1]
    br3 = [x + barWidth for x in br2]
    br4 = [x + barWidth for x in br3]
    br5 = [x + barWidth for x in br4]

    # Make the plot
    plt.bar(br1, lst_choice_hist[0], color=colors[6], width=barWidth,
            edgecolor='grey', label='e-greedy')
    plt.bar(br2, lst_choice_hist[1], color=colors[12], width=barWidth,
            edgecolor='grey', label='adaptive-greedy')
    plt.bar(br3, lst_choice_hist[2], color=colors[9], width=barWidth,
            edgecolor='grey', label='logistic_TS')
    plt.bar(br4, lst_choice_hist[3], color=colors[2], width=barWidth,
            edgecolor='grey', label='own_logistic_TS')
    plt.bar(br5, lst_choice_hist[4], color=colors[4],
            width=barWidth, label='real_observer')

    # Adding Xticks
    plt.xlabel('Bandits', fontweight='bold', fontsize=15)
    plt.ylabel('Number of choices', fontweight='bold', fontsize=15)
    plt.xticks([r + barWidth for r in range(len(lst_choice_hist[0]))],
               ['CB', 'STS', 'Speaker', 'non_Speaker', 'non_Speaker', 'non_Speaker'])

    plt.legend()
    plt.savefig(f'{outputs_dir}plot/{name_plot}.png')


def generate_plot(batch_size, lst_reward, y,  outputs_dir='outputs/train/'):

    def get_mean_reward(reward_lst, batch_size=batch_size):
        mean_rew = list()
        for r in range(len(reward_lst)):
            mean_rew.append(sum(reward_lst[:r+1]) * 1.0 / ((r+1)*batch_size))
        return mean_rew
    
    rcParams['figure.figsize'] = 20, 10
    lwd = 5
    cmap = plt.get_cmap('tab20')

    ax = plt.subplot(111)
    plt.plot(get_mean_reward(
        lst_reward[0]), label="Epsilon-Greedy (p0=20%, decay=0.9999)", linewidth=lwd, color=colors[6])  # rewards_egr
    plt.plot(get_mean_reward(
        lst_reward[1]), label="Logistic TS", linewidth=lwd, color=colors[9])  # rewards_lts
    plt.plot(get_mean_reward(
        lst_reward[2]), label="Adaptive Greedy (p0=30%, decaying percentile)", linewidth=lwd, color=colors[12])  # rewards_agr2

    # import warnings
    box = ax.get_position()
    """ ax.set_position([box.x0, box.y0 + box.height * 0.1,
                    box.width, box.height * 1.25]) """
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
              fancybox=True, ncol=3, prop={'size': 20})

    plt.tick_params(axis='both', which='major', labelsize=25)
    plt.xticks([i*50 for i in range(10)], [i*1000 for i in range(10)])

    plt.xlabel(
        f'Rounds (models were updated every {batch_size} rounds)', size=30)
    plt.ylabel('Cumulative Mean Reward', size=30)
    plt.title(
        f'Comparison of Online Contextual Bandit Policies\n(Base Algorithm is Logistic Regression)\n({y.shape[0]} categories, {y.shape[1]} attributes)', size=30)
    plt.grid()
    plt.savefig(f'{outputs_dir}plot/comparison_plot.png')


def generate_expected_reward(df,  outputs_dir='outputs/train/'):

    cum = df.reset_index().groupby(['patch_label'])['reward'].mean()
    plt.plot(cum.values,  linewidth=5)
    plt.title('Expected reward with {} simulation'.format(
        max(df.simul_id)+1), fontsize=10)
    plt.savefig(f'{outputs_dir}plot/ER_TS.png')


def comparison_ts_standScaler(df, n_dim=4,  outputs_dir='outputs/train/'):

    if (n_dim == 4):

        ts_m = {'m_1': df.iloc[0]['m_1'], 'm_id': df.iloc[0]['m_id'],
                'm_dist': df.iloc[0]['m_dist'], 'm_ang': df.iloc[0]['m_ang'], }
        coef_No_Scal_No_Int = {'m_1': df.iloc[1]['m_1'], 'm_id': df.iloc[1]
                               ['m_id'], 'm_dist': df.iloc[1]['m_dist'], 'm_ang': df.iloc[1]['m_ang'], }
        coef_No_Scal = {'m_1': df.iloc[2]['m_1'], 'm_id': df.iloc[2]['m_id'],
                        'm_dist': df.iloc[2]['m_dist'], 'm_ang': df.iloc[2]['m_ang'], }
        coef = {'m_1': df.iloc[3]['m_1'], 'm_id': df.iloc[3]['m_id'],
                'm_dist': df.iloc[3]['m_dist'], 'm_ang': df.iloc[3]['m_ang'], }

        fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(12, 4))
        ax1.bar(list(ts_m.keys()), list(ts_m.values()),
                color='r', width=0.2, edgecolor='grey')
        ax1.set_title('TS_m')

        ax2.bar(list(coef_No_Scal_No_Int.keys()), list(
            coef_No_Scal_No_Int.values()), color='g', width=0.2, edgecolor='grey')
        ax2.set_title('coef_No_Scal_No_Int')

        ax3.bar(list(coef_No_Scal.keys()), list(coef_No_Scal.values()),
                color='b', width=0.2, edgecolor='grey')
        ax3.set_title('coef_No_Scal')

        ax4.bar(list(coef.keys()), list(coef.values()),
                color='y', width=0.2, edgecolor='grey')
        ax4.set_title('coef')

        plt.savefig(f'{outputs_dir}plot/comparison_ts_stdScaler.png')

    else:
        ts_m = {'m_id': df.iloc[0]['m_id'], 'm_dist': df.iloc[0]
                ['m_dist'], 'm_ang': df.iloc[0]['m_ang'], }
        coef_No_Scal_No_Int = {
            'm_id': df.iloc[1]['m_id'], 'm_dist': df.iloc[1]['m_dist'], 'm_ang': df.iloc[1]['m_ang'], }
        coef_No_Scal = {
            'm_id': df.iloc[2]['m_id'], 'm_dist': df.iloc[2]['m_dist'], 'm_ang': df.iloc[2]['m_ang'], }
        coef = {'m_id': df.iloc[3]['m_id'], 'm_dist': df.iloc[3]
                ['m_dist'], 'm_ang': df.iloc[3]['m_ang'], }

        fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(12, 4))
        ax1.bar(list(ts_m.keys()), list(ts_m.values()),
                color='r', width=0.2, edgecolor='grey')
        ax1.set_title('TS_m')

        ax2.bar(list(coef_No_Scal_No_Int.keys()), list(
            coef_No_Scal_No_Int.values()), color='g', width=0.2, edgecolor='grey')
        ax2.set_title('coef_No_Scal_No_Int')

        ax3.bar(list(coef_No_Scal.keys()), list(coef_No_Scal.values()),
                color='b', width=0.2, edgecolor='grey')
        ax3.set_title('coef_No_Scal')

        ax4.bar(list(coef.keys()), list(coef.values()),
                color='y', width=0.2, edgecolor='grey')
        ax4.set_title('coef')

        plt.savefig(f'{outputs_dir}plot/comparison_ts_stdScaler.png')


def generate_kde_plot(results, vid_name, experiment,  outputs_dir):
    
    try:
        debug = config['STATS']['launch exp']
        phi = config ['TEST CONFIG']['phi']
        kappa = config ['TEST CONFIG']['kappa']


        real_vs_real = results[0]['mm']
        gen_vs_gen = results[1]['mm']
        gen_vs_real = results[2]['mm']

        fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))

        title = ['Vector', 'Direction', 'Length', 'Position', 'Duration']
        
        if debug:
            fig.suptitle(f'MultiMatch & ScanMatch - {vid_name} | phi : {phi} - K : {kappa}', fontsize=16)
        else:
            fig.suptitle(f'MultiMatch & ScanMatch - {vid_name}', fontsize=16)

        # loop through each dimension
        for i in range(3):
            sns.kdeplot(real_vs_real[:, i], fill=True,
                        label='Real vs Real', ax=axes[0, i])
            sns.kdeplot(gen_vs_gen[:, i], fill=True,
                        label='Gen vs Gen', ax=axes[0, i])
            sns.kdeplot(gen_vs_real[:, i], fill=True,
                        label='Gen vs Real', ax=axes[0, i])
            axes[0, i].legend(loc='upper right')
            axes[0, i].set_title(title[i])

        # loop through additional dimensions
        for i in range(2):
            sns.kdeplot(real_vs_real[:, i+3], fill=True,
                        label='Real vs Real', ax=axes[1, i])
            sns.kdeplot(gen_vs_gen[:, i+3], fill=True,
                        label='Gen vs Gen', ax=axes[1, i])
            sns.kdeplot(gen_vs_real[:, i+3], fill=True,
                        label='Gen vs Real', ax=axes[1, i])
            axes[1, i].legend(title=title[i+3], loc='upper right')
            axes[1, i].set_title(title[i+3])

        # remove the last column in the second row
        # axes[1,2].remove()
        real_vs_real = results[0]['sm']
        gen_vs_gen = results[1]['sm']
        gen_vs_real = results[2]['sm']
        
        

        sns.kdeplot(np.squeeze(real_vs_real), fill=True, )
        sns.kdeplot(np.squeeze(gen_vs_gen), fill=True, )
        sns.kdeplot(np.squeeze(gen_vs_real), fill=True, )
        plt.legend(loc='upper right', labels=[
                'Real vs Real', 'Gen vs Gen', 'Gen vs Real'])
        plt.title('ScanMatch Distribution')
        plt.savefig(f'{outputs_dir}stats_match.png')
        result_dir = config['STATS']['result dir']
        plt.savefig(f'{result_dir}stats_match{experiment}.png')

        
        # Debug Results 
        if config['STATS']['launch exp'] :
            result_dir = config['STATS']['result dir']
            experiment = config['TEST CONFIG']['experiment']

            plt.savefig(f'{result_dir}stats_match{experiment}.png')

    except Exception as e:
        print(e)


def generate_exponential_Q(all_gen, outputs_dir, vid_name,  start_frame=0, end_frame=50, save_all_obs=False):
    
    gif_folder = f'{outputs_dir}GIF_exp_Q/'
    os.makedirs(gif_folder, exist_ok=True)
    
    def generate_gif(gen):
        # gp, curr_fix_dur, curr_rho, A, Q, dist, jump_prob
        gp = gen[:][:, 0]
        curr_fix_dur = gen[:][:, 1]
        curr_rho = gen[:][:, 2]
        A = gen[:][:, 3]
        Q = gen[:][:, 4]
        dist = gen[:][:, 5]
        jump_prob = gen[:][:, 6]

        array = [0]

        # create the figure and axis objects
        fig, ax = plt.subplots()
        ax.set_ylim([0, 1])
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        line, = ax.plot([], [])
        hline = ax.axhline(0, linestyle="--", color="red")
        hline_label = ax.text(2, 2, "", ha="right",
                                va="bottom", color='red', fontweight='bold')
        vline = ax.axvline(0, 0, 1, linestyle="--", color="green", )
        vline_label = ax.text(2, 2, "", ha="left",
                                va="top", color='green',  fontweight='bold')

        # function to generate data for the plot
        def inv_exp(gp, frame,):
            ax.set_xlim([0, frame*2+1])
            x = np.linspace(0, frame, 100)
            a = 1 / (1 + np.exp(-1 * (gp - 1)))
            b = 1 / (1 + np.exp(-1 * gp))
            y = a + (b - a) * np.exp(-x)

            if len(y) > 0:
                x_max = frame*2
                y_max = y[-1]
                x = np.concatenate([x, [x_max]])
                y = np.concatenate([y, [y_max]])

            # y = np.maximum(y, Q)  # set lower bound
            return x, y

        # function to update the plot
        def update(frame):

            if curr_fix_dur[frame] == 0:
                array.append(frame)

            x, y = inv_exp(gp=gp[frame], frame=frame)
            line.set_data(x, y)
            ax.set_title(
                f" Frame={frame} GP={gp[frame]:.2f}  Q={Q[frame]:.2f} dist={dist[frame]:.2f} JUMP={jump_prob[frame]:.2f}")

            # update horizontal red line
            hline.set_ydata([Q[frame], Q[frame]])
            hline_label.set_text(f"Q = {Q[frame]:.2f}")
            hline_label.set_position((frame - 2, Q[frame]))

            # update vertical green line
            vline.set_ydata([y[-1], Q[frame]])
            vline.set_xdata([frame, frame])
            vline_label.set_text(f"y = {y[-1]:.2f}")
            vline_label.set_position((frame + 2, y[-1]))

            return line, hline, hline_label, vline, vline_label

        anim = FuncAnimation(fig, update, frames=range(end_frame))
        return anim

   
    for obs in range(len(all_gen)):
        anim = generate_gif(all_gen[obs])
        anim.save(f'{outputs_dir}GIF_exp_Q/obs_{obs}.gif', writer='imagemagick')
        
    """if not save_all_obs :
        shutil.rmtree(gif_folder) """


def create_distribution_duration(real_scans, gen_dur, n_real, experiment, outputs_dir):
    
    phi = config ['TEST CONFIG']['phi']
    kappa = config ['TEST CONFIG']['kappa']
    
    n_cols = 5  # Set the number of columns for the subplots
    # Calculate the number of rows required based on n_real
    n_rows = math.ceil(n_real / n_cols)
    # Create a figure with nrows and ncols and adjust the size as needed
    fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols,
                            figsize=(20, 4 * n_rows))
    
    fig.suptitle(f'Duration Plot  | phi : {phi} - K : {kappa}', fontsize=16)

    plt.subplots_adjust(top=0.9, bottom=0.2)

    for i, ax in enumerate(axs.flat):  # Loop over the subplots
        # Get the data for the i-th subplot if it exists, otherwise set an empty list
        real_data = real_scans[i][0][:, 2] if i < n_real else []
        gen_data = gen_dur[i] if i < len(gen_dur) else []
        # Plot the data on the i-th subplot
        sns.kdeplot(real_data, fill=False, ax=ax)
        # Plot the data on the i-th subplot
        sns.kdeplot(gen_data, fill=False, ax=ax)

        ax.set_xlabel('Value')  # Set the x-axis label for the i-th subplot
        ax.set_ylabel('Density')  # Set the y-axis label for the i-th subplot
        ax.legend(labels=['REAL duration', 'GEN duration'], loc='upper right')
        # Set the title for the i-th subplot
        ax.set_title(f'Distribution of the Observer: {i+1}')

    plt.tight_layout()  # Adjust the layout of the subplots to prevent overlap
    # Save the figure to a file
    plt.savefig(f'{outputs_dir}fix_duration.png')
    result_dir = config['STATS']['result dir']
    plt.savefig(f'{result_dir}fix_duration.png')
    
    if config['STATS']['launch exp'] :
        result_dir = config['STATS']['result dir'] 
        experiment = config['TEST CONFIG']['experiment']
        plt.savefig(f'{result_dir}fix_duration.png')
    
    
    fig, ax = plt.subplots(figsize=(8,6))

    # Plot the real data
    real_data = np.concatenate([scan[0][:, 2] for scan in real_scans])
    sns.kdeplot(real_data, fill=False,  ax=ax)

    # Plot the generated data
    gen_data = np.concatenate(gen_dur)
    sns.kdeplot(gen_data, fill=False, ax=ax)
    
    np.save(f'{result_dir}real.npy', real_data)
    np.save(f'{result_dir}gen.npy', gen_data)
    
    ax.set_xlabel('Value')
    ax.set_ylabel('Density')
    ax.legend(labels=['Real duration', 'Generated duration'])
    ax.set_title(f'Distribution of Observers | Phi: {phi} | Kappa: {kappa} ')

    plt.tight_layout()
    plt.savefig(f'{outputs_dir}full_fix_duration.png')
    result_dir = config['STATS']['result dir']

    plt.savefig(f'{result_dir}full_fix_duration.png')

    
    # Debug Results 
    if config['STATS']['launch exp'] :
        result_dir = config['STATS']['result dir'] 
        experiment = config['TEST CONFIG']['experiment']

        plt.savefig(f'{result_dir}full_fix_duration.png')


def create_wide_saccade(real_amps, gen_amps, n_real, experiment,outputs_dir):

    #Plot Saccade amplitude Distributions
    plt.figure()
    sns.kdeplot(real_amps, fill=False)
    sns.kdeplot(gen_amps, fill=False)
    plt.legend(title='Saccades Amplitude Distributions', loc='upper right', labels=['Real', 'Generated'])

    plt.savefig(f'{outputs_dir}amplitude_saccade.png')
    result_dir = config['STATS']['result dir']
    plt.savefig(f'{result_dir}amplitude_saccade.png')
    
    if config['STATS']['launch exp'] :
        result_dir = config['STATS']['result dir'] 
        experiment = config['TEST CONFIG']['experiment']
        plt.savefig(f'{result_dir}amplitude_saccade.png')
    


def create_direction_saccade(real_dirs, gen_dirs, experiment, outputs_dir):
    #Plot Saccade Direction Distribution
    fig = plt.figure()
    bin_size = 10
    s_dir_rad = real_dirs
    s_dir_deg = np.rad2deg(s_dir_rad)
    a,b = np.histogram(s_dir_deg, bins=np.arange(0, 360+bin_size, bin_size))
    centers = np.deg2rad(np.ediff1d(b)//2 + b[:-1])
    ax3 = fig.add_subplot(1,2,1, projection='polar')
    ax3.bar(centers, a, width=np.deg2rad(bin_size), bottom=0.0, color='blue' )
    ax3.set_title('Real Saccades Direction')
    s_dir_rad = gen_dirs
    s_dir_deg = np.rad2deg(s_dir_rad)
    a,b = np.histogram(s_dir_deg, bins=np.arange(0, 360+bin_size, bin_size))
    centers = np.deg2rad(np.ediff1d(b)//2 + b[:-1])
    ax3 = fig.add_subplot(1,2,2, projection='polar')
    ax3.bar(centers, a, width=np.deg2rad(bin_size), bottom=0.0, color='orange')
    ax3.set_title('Gen Saccades Direction')
    
    plt.savefig(f'{outputs_dir}direction_saccade.png')
    result_dir = config['STATS']['result dir']
    plt.savefig(f'{result_dir}direction_saccade.png')


    