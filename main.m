clc;close all;clear all;
rng(77316669) % fixing a random seed for selection
output_on = true; % controls the output of printf

heart_data = load("cleveland_heart_data.txt");
heart_data(:, 14) = heart_data(:, 14) > 0; % converts 1,2,3,4 -> 1 and 0 -> 0

N = size(heart_data, 1);
training_data_len = 200;
shuffled_idx = randperm(N);

training_data = heart_data(shuffled_idx(1:training_data_len), :);
test_data = heart_data(shuffled_idx(training_data_len+1:end), :);

% Helpers
counter = @(column,value)(sum(column == value,"all"));
printf = @(fmt,varargin) (output_on && fprintf(fmt + "\n",varargin{:}));
get_state = @(data) 16*data(:,1) + 4*(data(:,2)-1) + 2*data(:,3) + data(:,4) + 1; % generates a unique state (5-bit binary) for part 2.c

% PART 2 - A classifier based on a joint conditional PMF
reduced_training_data = training_data(:,[2,3,6,9,14]); % sex, cp, fbs, exang, disease

D_train = reduced_training_data(:, 5); % 5th column of reduced_training_data is target D

n0 = counter(D_train, 0);
n1 = counter(D_train, 1);

P_D0 = n0 / training_data_len;
P_D1 = n1 / training_data_len;

printf("\n--- Part 2.a: Class Prior Probablities ---");

printf("P(D=0) = %.3f (n0 = %3d)", P_D0, n0);
printf("P(D=1) = %.3f (n1 = %3d)", P_D1, n1);

% 2.c - Class conditional PMFs
train_states = get_state(reduced_training_data);
S = 32;
n_x0 = zeros(S, 1);
n_x1 = zeros(S, 1);

for x = 1:S
    n_x0(x) = sum(train_states == x & D_train == 0);
    n_x1(x) = sum(train_states == x & D_train == 1);
end
% Conditional PMFs P(X = x | D = d)
P_X_given_D0 = n_x0 / n0;
P_X_given_D1 = n_x1 / n1;

% Print the Conditional PMFs
printf("\n--- Part 2.c: Conditional PMFs ---");
printf("\n%-5s | %-3s %-2s %-3s %-5s | %-5s %-12s | %-5s %-12s", ...
    "State", "sex", "cp", "fbs", "exang", "n_x0", "P(X=x|D=0)", "n_x1", "P(X=x|D=1)");
printf("%s", repmat('-', 1, 68));

for x = 1:S
    sex_val   = floor((x - 1) / 16);
    cp_val    = floor(mod(x - 1, 16) / 4) + 1;
    fbs_val   = floor(mod(x - 1, 4) / 2);
    exang_val = mod(x - 1, 2);

    printf("%5d | %3d %2d %3d %5d | %5d %12.4f | %5d %12.4f", ...
        x, sex_val, cp_val, fbs_val, exang_val, ...
        n_x0(x), P_X_given_D0(x), n_x1(x), P_X_given_D1(x));
end

printf("%s", repmat('-', 1, 68));
printf("%-24s | %5d %12.4f | %5d %12.4f", ...
    "Total", sum(n_x0), sum(P_X_given_D0), sum(n_x1), sum(P_X_given_D1));

% 2.d - Laplace-smoothed conditional PMFs (alpha = 1)
alpha = 1;
P_X_given_D0_smooth = (n_x0 + alpha) / (n0 + alpha * S);
P_X_given_D1_smooth = (n_x1 + alpha) / (n1 + alpha * S);

printf("\n--- Part 2.d: Laplace-smoothed Conditional PMFs ---");
% Print smoothed PMFs table and verify normalization
printf("\n%-5s | %-3s %-2s %-3s %-5s | %-5s %-14s | %-5s %-14s", ...
    "State", "sex", "cp", "fbs", "exang", "n_x0", "P_hat(X=x|D=0)", "n_x1", "P_hat(X=x|D=1)");
printf("%s", repmat('-', 1, 72));
for x = 1:S
    sex_val   = floor((x - 1) / 16);
    cp_val    = floor(mod(x - 1, 16) / 4) + 1;
    fbs_val   = floor(mod(x - 1, 4) / 2);
    exang_val = mod(x - 1, 2);
    printf("%5d | %3d %2d %3d %5d | %5d %14.4f | %5d %14.4f", ...
        x, sex_val, cp_val, fbs_val, exang_val, ...
        n_x0(x), P_X_given_D0_smooth(x), n_x1(x), P_X_given_D1_smooth(x));
end
printf("%s", repmat('-', 1, 72));
printf("%-24s | %5d %14.4f | %5d %14.4f", ...
    "Total", sum(n_x0), sum(P_X_given_D0_smooth), sum(n_x1), sum(P_X_given_D1_smooth));

% 2.f - Evaluate classifier on test set
reduced_test_data = test_data(:, [2, 3, 6, 9, 14]); % sex, cp, fbs, exang, disease
D_test = reduced_test_data(:, 5);
test_states = get_state(reduced_test_data);
% Compute posterior scores P_hat(D = d) * P_hat(X = x | D = d) for each test patient
score_D0 = P_D0 * P_X_given_D0_smooth(test_states);
score_D1 = P_D1 * P_X_given_D1_smooth(test_states);
% Predict D = 1 if score_D1 >= score_D0, else 0
D_pred = double(score_D1 >= score_D0);
% Accuracy
num_test = length(D_test);
num_correct = sum(D_pred == D_test);
accuracy_2f = num_correct / num_test;
% 2x2 Confusion Matrix (Rows: True D=0,1 | Cols: Pred D=0,1)
conf_mat_2f = [
    sum(D_test == 0 & D_pred == 0), sum(D_test == 0 & D_pred == 1);
    sum(D_test == 1 & D_pred == 0), sum(D_test == 1 & D_pred == 1)
];
printf("\n--- Part 2.f: Joint Conditional PMF Classifier Results ---");
printf("Test Accuracy: %.4f (%d / %d) [%.2f%%]", accuracy_2f, num_correct, num_test, accuracy_2f * 100);
printf("\nConfusion Matrix (Rows: True D, Cols: Predicted D):");
printf("%12s | %10s %10s", "", "Pred D=0", "Pred D=1");
printf("%s", repmat('-', 1, 36));
printf("%12s | %10d %10d", "True D=0", conf_mat_2f(1,1), conf_mat_2f(1,2));
printf("%12s | %10d %10d", "True D=1", conf_mat_2f(2,1), conf_mat_2f(2,2));