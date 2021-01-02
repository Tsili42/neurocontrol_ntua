% system dynamics
A = [0 1 0; 0 0 1; 0 0 0];
B = [0; 0; 1];

% LQR args
M = eye(3);
R = 42; % rho doesnt change the solution for k

% Q learning matrices
x = zeros([3 11]);
u = rand([10 1]); % random input values

for i=2:11
    x(:,i) = A*x(:,i-1)+B*u(i-1); % samples taken from system
end

Z = zeros(10);

for i=1:10
    Z(i,:) = [x(1,i)^2 x(2,i)^2 x(3,i)^2 x(1,i)*x(2,i) x(2,i)*x(3,i) x(1,i)*x(3,i) u(i)^2 2*u(i)*x(1,i) 2*u(i)*x(2,i) 2*u(i)*x(3,i)];
end

if rank(Z) < 10
    return
end

% Start of iterative procedure
H_old = zeros(4); % 1st approximaton is chosen, arbitrarily, as zero(4x4)
H_new = zeros(4);

k_old = zeros([1 3]);
k_new = rand([1 3]); % no reason to be random, just for technical reasons in order to get inside the while loop the 1st time

d = zeros([10 1]);
Z_inv = inv(Z); % calculate once for speed
while (abs(norm(k_old-k_new)) > 0.1) % set here desired value of approximation
    H_old = H_new;
    k_old = k_new;
    
    for i=1:10
        d(i,:) = (x(:,i)')*x(:,i)+R*u(i)^2+[x(:,i+1)' x(:,i+1)'*k_old']*H_old*[x(:,i+1)' x(:,i+1)'*k_old']';
    end
    
    temp = Z_inv*d;
    
    H_new(1,1) = temp(1,1);
    H_new(1,2) = temp(4,1)/2; % Matrix M of cost function is symmetrical so H will also be symmetrical
    H_new(1,3) = temp(6,1)/2;
    H_new(1,4) = temp(8,1);
    H_new(2,1) = H_new(1,2); 
    H_new(2,2) = temp(2,1);
    H_new(2,3) = temp(5,1)/2;
    H_new(2,4) = temp(9,1);
    H_new(3,1) = H_new(1,3);
    H_new(3,2) = H_new(2,3);
    H_new(3,3) = temp(3,1);
    H_new(3,4) = temp(10,1);
    H_new(4,1) = H_new(1,4);
    H_new(4,2) = H_new(2,4);
    H_new(4,3) = H_new(3,4);
    H_new(4,4) = temp(7,1);
    
    k_new = (H_new(4,4)^(-1))*H_new(4,1:3);
end

H_new, k_new

t = 1:50;
test_x = zeros([3 50]);
for i=1:11
    test_x(:,i) = x(:,i);
end
for i=12:50
    test_x(:,i) = A*test_x(:,i-1)+B*k_new*test_x(:,i-1);
end

fig1 = figure;  
subplot(3,1,1); 
plot(t,test_x(1,:)); 
ylabel('x_{1}'); 
xlabel('time t');  

subplot(3,1,2); 
plot(t,test_x(2,:)); 
ylabel('x_{2}'); 
xlabel('time t');  

subplot(3,1,3); 
plot(t,test_x(3,:)); 
ylabel('x_{3}'); 
xlabel('time t');