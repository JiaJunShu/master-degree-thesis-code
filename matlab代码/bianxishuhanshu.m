% ����x�᷶Χ
x = -50:0.1:50;

% ���庯��
y1 = 1 ./ (1 + exp(-x));
y2 = 1 ./ (1 + exp(-x/10));
y3 = 1 ./ (1 + exp(-x/20));
y4 = 1 ./ (1 + exp(-x/50));

% ����ͼ��
plot(x, y1, 'k-', 'LineWidth', 2); hold on;
plot(x, y2, 'k--', 'LineWidth', 2);
plot(x, y3, 'k-.', 'LineWidth', 2);
%plot(x, y4, 'k:', 'LineWidth', 2);
hold off;

set(gcf, 'Color', 'w'); % ���ñ�����ɫΪ��ɫ

% ���ͼ������ע
legend('k4=1', ...
       'k4=10', ...
       'k4=20', ...
       'k4=50');
xlabel('��x');
ylabel('sigmoid����');
%title('���� y = 1 / (1 + exp(x)) �������Ƚ�ͼ');
%grid on;
