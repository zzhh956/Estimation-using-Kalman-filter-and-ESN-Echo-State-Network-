%Create data
data = table2array(IFNNY(:, 5));
for i = 1: 5000
    data(end + 1) = data(end) + (rand - 0.5) * tmp * 0.1;
end

%ESN 
inputData = data(1: end - 1); 
targetData = data(2: end);

washout = 10;
trlen = 4000; tslen = 200;

trX{1} = inputData(1: trlen);
tsX{1} = inputData(1 + trlen: trlen + tslen);

trY = targetData(1 + washout: trlen);
tsY = targetData(1 + trlen + washout: trlen + tslen);

esn = ESN(100, 'leakRate', 0.3, 'spectralRadius', 0.05, 'regularization', 1e-8);

esn.train(trX, trY, washout);

output = esn.predict(tsX, washout);

%讀取資料並設定遞迴次數
iteration = 100;
N = length(tsY);
data_kalman = tsY;

%設定kalman filter的參數
Q = 10^(-2);
P = Q;
F = 1;
R = 1;
H = 1;

%假設measurement的結果
v_variance = 1;
noise_v = sqrt(v_variance) * randn(N, 1);
z = H * data_kalman + noise_v;

%Kalman filter
x_ite = data_kalman(:, 1);
for ite = 1: iteration
    %prior estimation
    x_ite = (F * x_ite')';
    P_ite = F^2 * P + Q;
    
    %gain
    K = P * H / (H^2 * P + R);
    
    %posterior estimation
    y = z - (H * x_ite);
    x_ite = x_ite + y * K;
    P_ite = (1 - K * H) * P_ite;
end

%error
error = immse(output, tsY);
fprintf('Test error of ESN: %g\n', error);
error = immse(x_ite, tsY);
fprintf('Test error of Kalman Filter: %g\n', error);

%plot
figure(1)
plot(1: N, tsY, 'b', 1: N, x_ite, 'g', 1:length(output), output, 'r');

title("price prediction");
xlabel("number of data");
ylabel("price");
legend("real price", "Kalman", "ESN");