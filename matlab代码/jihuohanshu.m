% ����x�ķ�Χ
x = -3:0.1:3;

% ����ReLU����
relu = max(0, x);

% ����Tanh����
tanh_func = tanh(x);

% ����Sigmoid����
sigmoid = 1 ./ (1 + exp(-x));

% ����ͼ��
figure;
set(gcf, 'Color', 'w'); % ����ͼ�δ��ڵı���Ϊ��ɫ
plot(x, relu, '-k', 'LineWidth', 2); % ReLU������ʵ��
hold on; % ���ֵ�ǰͼ��
plot(x, tanh_func, '--k', 'LineWidth', 2); % Tanh����������
plot(x, sigmoid, ':k', 'LineWidth', 2); % Sigmoid����������

% ����ͼ��
legend('ReLU', 'Tanh', 'Sigmoid');

% ���ӱ�����������ǩ
%title('�����');
xlabel('x');
ylabel('y');

% ����������


% ��ʾͼ��
hold off;
