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