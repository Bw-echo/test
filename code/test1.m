% 风险敏感残差发生器 + LQR+风险敏感残差补偿控制器
clear; clc; close all;

%% 1. 系统参数定义
Ts = 1/20000;
Lf = 2e-3;
rf = 0.01;
Cf = 1500e-6;
w0 = 2*pi*50;

Ac = [-rf/Lf,  w0,    -1/Lf,   0;
      -w0,    -rf/Lf,  0,     -1/Lf;
       1/Cf,   0,      1,      w0;
       0,      1/Cf,  -w0,     1];

Bc = [1/Lf,0,0,0;
      0,1/Lf,0,0;
      0,0,1/Cf,0;
      0,0,0,1/Cf];

C = eye(4);
D = zeros(4,4);

sysd = c2d(ss(Ac,Bc,C,D),Ts,'zoh');
A = sysd.A;B = sysd.B;C = sysd.C;D = sysd.D;
nx = size(A,1);nu = size(B,2);ny = size(C,1);


%% 2. 风险敏感残差发生器
Qw = 0.01*eye(nx);     % 过程噪声协方差
Rv = 0.01*eye(ny);     % 测量噪声协方差
tau = 1;               % τ-散度参数
theta_k = -0.15;       % 风险敏感因子（负值，惩罚不确定性）
maxIter = 500;         % 最大迭代次数
tol = 1e-10;           % 收敛容差 

Pk = Qw;
Vk = Pk;
Lk = zeros(nx,ny);

for iter = 1:maxIter
    %  预测协方差 (9)：P_bar = A*Vk*A' ）
    Pbar = A*Vk*A' + Qw;       
    % 确保对称正定
    Pbar = (Pbar + Pbar')/2;   
    Lp = chol(Pbar + 1e-12*eye(nx),'lower');
    % 根据 τ 更新 Vk，(16)
    if tau == 1
        Vk_new = Lp * expm(theta_k*(Lp'*Lp)) * Lp';
    elseif tau > 0 && tau < 1
        M = eye(nx) - theta_k*(1-tau)*(Lp'*Lp);
        Vk_new = Lp * (M^(1/(tau-1))) * Lp';
    else
        error('tau 只能取 1 或 0<tau<1');
    end

    Vk_new = (Vk_new + Vk_new')/2;

    % 计算滤波器增益 Lk，(19)
    S = C*Vk_new*C' + Rv;
    S = (S + S')/2;
    Lk_new = A*Vk_new*C' / S;  % 维度修正：Lk 应为 nx × ny
   
    % 后验协方差(18) 
   Pk_new = Pbar - Lk_new*S*Lk_new'+B*B';
    % Pk_new = Lp*Lp';
    Pk_new = (Pk_new + Pk_new')/2;

    if norm(Pk_new-Pk,'fro') < tol
        Pk = Pk_new;
        Vk = Vk_new;
        Lk = Lk_new;
        break;
    end
    Pk = Pk_new;
    Vk = Vk_new;
    Lk = Lk_new;
end

fprintf('Lk solved, iter = %d, max eig(A-Lk*C)=%.6f\n', ...
    iter, max(abs(eig(A-Lk*C))));

%% 3.状态反馈控制 (LQR,F)
% 定义LQR权重矩阵
QF = diag([1 1 20 20]);       %状态代价权重
RF = diag([1e6 1e6 10 10]);   %控制代价权重
% 状态反馈控制律u_F = F*xhat，u=-Kx  满足A+B*F稳定 
[Kdlqr,~,~] = dlqr(A,B,QF,RF);
F = -Kdlqr;     

fprintf('F solved, max eig(A+B*F)=%.6f\n', ...
    max(abs(eig(A+B*F))));

%% 4. 广义系统 P(z)
% 状态向量: xP = [xrd; xd; xT] 
% x_rd: 残差发生器状态（估计误差） 
% x_d: 标称被控对象状态
% x_T: 性能加权状态
% 输入:w = dl(扰动);输出:u = uc(补偿信号)
% output z = [sqrt(We)*(xd+xT); sqrt(Wu)*uc], measurement y = residual r = C*xrd
Ard = A - Lk*C;

A_aug = blkdiag(Ard, A, A);               %A,3*nx 维
B1_aug = [B; B; zeros(nx,nu)];            % 扰动输入通道dl作用于x_rd和x_d
B2_aug = [zeros(nx,nu); zeros(nx,nu); B]; % 补偿控制uc作用于x_T

% 性能输出 z: 包含电压跟踪误差
We = diag([0.1 0.1 10 10]);       %状态误差权重
Wu = diag([1e-6 1e-6 1 1]);       %补偿控制量权重
% Swe = sqrtm(We); Swu = sqrtm(Wu);
% z = We*(C*x_d + C*x_T)
Cz_aug = [zeros(ny,nx), We*C, We*C];
Cr_aug = [C, zeros(ny,2*nx)];

Dz1 = zeros(ny,nu);
Dz2 = Wu;

Dr1 = zeros(ny,nu);
Dr2 = zeros(ny,nu);
% 组装广义系统 P
P = ss(A_aug, [B1_aug, B2_aug], ...
       [Cz_aug; Cr_aug], ...
       [Dz1, Dz2; Dr1, Dr2], Ts);

%% 5. H∞ / 风险敏感补偿控制器
% theta = -gamma^(-2)，把指数风险代价转化为 Hinf/鲁棒问题。
gamma = 0.6;         % 初始界（可调，需要保证解存在0<gamma<1）
theta = -1/gamma^2;

nmeas = ny;      % 测量输入为残差 r
ncont = nu;      % 控制输出为补偿信号 u_c

opts = hinfsynOptions('Method','RIC','Display','off');
[Qz,~,gammaAchieved] = hinfsyn(P,nmeas,ncont,[0,gamma],opts);
fprintf('H∞补偿器完成，实际 gamma = %.6g\n', gammaAchieved);

[Aq,Bq,Cq,Dq] = ssdata(Qz);
fprintf('补偿控制器 Q(z) 阶数 = %d\n', size(Aq,1));
 
% if gammaAchieved <= gamma
%     fprintf('H∞控制器设计成功，实际γ = %.6g (≤ %.6g)\n', gammaAchieved, gamma);
% else
%     warning('无法满足 gamma_fixed=%.6g，实际为 %.6g，请增大 gamma_fixed', gamma, gammaAchieved);
% end
% 
% [Aq, Bq, Cq, Dq] = ssdata(Qz);
% fprintf('补偿控制器 Q(z) 阶数 = %d\n', size(Aq,1));

%% 6. 仿真设置
N = 40000;
t = (0:N-1)'*Ts;

u_od = 25;            %期望参考值
x_ref = [5;0;25;0];   

% 稳态前馈：使 x_ref = A*x_ref + B*u_ref
u_ref = B \ ((eye(nx)-A)*x_ref);

% 负载扰动
dload = zeros(nu,N);
dload(3,round(0.01/Ts):end) = -4;
dload(4,:) = 0;

% 模型不确定性：LC 参数摄动（真实系统中 Lf 和 Cf 偏离标称值）
Lf_act = Lf * 1;     % 电感
Cf_act = Cf * 1;     % 电容

% 实际系统参数
Ac_act = [-rf/Lf_act,  w0,        -1/Lf_act,   0;
          -w0,        -rf/Lf_act,  0,         -1/Lf_act;
           1/Cf_act,   0,          1,          w0;
           0,          1/Cf_act,  -w0,         1];

Bc_act = [1/Lf_act,0,0,0;
          0,1/Lf_act,0,0;
          0,0,1/Cf_act,0;
          0,0,0,1/Cf_act];

sys_act = c2d(ss(Ac_act,Bc_act,C,D),Ts,'zoh');
A_act = sys_act.A;
B_act = sys_act.B;

% 1.无扰动
x_nodist = x_ref;
y_nodist = zeros(N,ny);

for k = 1:N
    x_nodist = A_act*x_nodist + B_act*u_ref;
    y_nodist(k,:) = (C*x_nodist)';
end

% 2.有扰动
x_open = x_ref;
y_open = zeros(N,ny);

for k = 1:N
    u_act = u_ref + dload(:,k);
    x_open = A_act*x_open + B_act*u_act;
    y_open(k,:) = (C*x_open)';
end

% 3.加入 LQR 
x_no = x_ref;
xhat_no = x_ref;

y_no = zeros(N,ny);
r_no = zeros(N,ny);

% 积分器：只积分 uod、uoq 两个电压误差
eint_no = zeros(2,1);

% 积分增益，可继续微调
Ki_v = diag([80,80]);

for k = 1:N
    y = C*x_no + 0.02*randn(ny,1);

    r = y - C*xhat_no;

    voltage_error = [x_ref(3)-y(3);
                     x_ref(4)-y(4)];

    eint_no = eint_no + Ts*voltage_error;

    % 防积分饱和
    eint_no = max(min(eint_no,10),-10);

    ui = [0;0;Ki_v*eint_no];

    x_error = xhat_no - x_ref;

    u = u_ref + F*x_error + ui;

    xhat_no = A*xhat_no + B*u + Lk*r;
    x_no = A_act*x_no + B_act*(u + dload(:,k));

    y_no(k,:) = (C*x_no)';
    r_no(k,:) = r';
end

% 4.LQR + 残差补偿
x_comp = x_ref;
xhat_comp = x_ref;
xq = zeros(size(Aq,1),1);

y_comp = zeros(N,ny);
r_comp = zeros(N,ny);

eint_comp = zeros(2,1);

for k = 1:N
    y = C*x_comp + 0.02*randn(ny,1);
    r = y - C*xhat_comp;
    voltage_error = [x_ref(3)-y(3);
                     x_ref(4)-y(4)];
    eint_comp = eint_comp + Ts*voltage_error;
    eint_comp = max(min(eint_comp,10),-10);
    ui = [0;0;Ki_v*eint_comp];
    x_error = xhat_comp - x_ref;
    uc = Cq*xq + Dq*r;
    % 可限制补偿量，避免 H∞ 控制器输出过大
    uc = max(min(uc,2),-2);
    u = u_ref + F*x_error + ui + uc;
    xq = Aq*xq + Bq*r;
    xhat_comp = A*xhat_comp + B*u + Lk*r;
    x_comp = A_act*x_comp + B_act*(u + dload(:,k));
    y_comp(k,:) = (C*x_comp)';
    r_comp(k,:) = r';
end

%% 7.性能统计
idx = round(0.05/Ts):N;

mean_nodist = mean(y_nodist(idx,3));  % u_od
mean_open   = mean(y_open(idx,3));
mean_no     = mean(y_no(idx,3));
mean_comp   = mean(y_comp(idx,3));

rms_nodist = rms(y_nodist(idx,3));
rms_open   = rms(y_open(idx,3));
rms_no     = rms(y_no(idx,3));
rms_comp   = rms(y_comp(idx,3));

dev_nodist = rms_nodist - u_od;
dev_open   = rms_open - u_od;
dev_no     = rms_no - u_od;
dev_comp   = rms_comp - u_od;

ratio = (1 - abs(dev_comp)/abs(dev_no))*100;

fprintf('\n========== 仿真结果(含参数摄动) ==========\n');
fprintf('theta = %.6g, gamma = %.6g, 参考电压 u_od = %.6g\n', theta, gamma, u_od);
fprintf('--------------------------------------------------------------------------\n');
fprintf(' %-24s | %12s | %12s\n', '控制策略', 'RMS(V)', '稳态偏差(V)');
fprintf('--------------------------------------------------------------------------\n');
fprintf(' %-24s | %12.6f | %+12.6f\n', '期望电压', u_od, 0);
fprintf(' %-24s | %12.6f | %+12.6f\n', '无扰动', rms_nodist, dev_nodist);
fprintf(' %-24s | %12.6f | %+12.6f\n', '有扰动', rms_open, dev_open);
fprintf(' %-24s | %12.6f | %+12.6f\n', 'LQR ', rms_no, dev_no);
fprintf(' %-24s | %12.6f | %+12.6f\n', 'LQR 残差补偿', rms_comp, dev_comp);
fprintf('--------------------------------------------------------------------------\n');
fprintf(' 补偿改善率 (相对于仅LQR)：%.2f%%\n', ratio);
fprintf('==========================================================================\n');

%% 12. 绘图
figure('Name','稳态误差消除效果');

subplot(2,1,1);
plot(t,y_nodist(:,3),'m-', ...
     t,y_open(:,3),'k-', ...
     t,y_no(:,3),'g--', ...
     t,y_comp(:,3),'r-', ...
     'LineWidth',1);
yline(5,'k:','LineWidth',1);
xlabel('时间/s');
ylabel('u_{od}/V');
grid on;
legend('无扰动','有扰动无控制','LQR+积分','LQR+积分+残差补偿','Location','best');
title('d轴电容电压对比');

subplot(2,1,2);
plot(t,r_no(:,1),'b--',t,r_comp(:,1),'r-','LineWidth',1);
xlabel('时间/s');
ylabel('残差 r_1');
grid on;
legend('LQR+积分','LQR+积分+残差补偿','Location','best');
title('残差信号对比');