% ��ȡ XLSX �ļ�
data = xlsread('k4error.xlsx');
x = data(:, 4);  % ������Ϊ������
y1 = data(:, 1); % ��һ��Ϊ������1
y2 = data(:, 5); % ������Ϊ������2
y3 = data(:, 6); % ������Ϊ������3

% ����ʽ��ϣ����� 15 �׶���ʽ��
p1 = polyfit(x, y1, 5);  % ��һ���������
p2 = polyfit(x, y2, 5);  % �������������
p3 = polyfit(x, y3, 5);  % �������������

% ʹ����ϲ�������Ԥ��ֵ
y1_fit = polyval(p1, x);
y2_fit = polyval(p2, x);
y3_fit = polyval(p3, x);

% ����ͼ�δ���
figure;
set(gcf, 'Color', 'w');  % ���ñ�����ɫΪ��ɫ

% ���Ƶ�һ�����ݼ��������
%plot(x, y1, 'ok', 'DisplayName', '��һ������');  % ԭʼ����ɢ��ͼ
hold on;
plot(x, y1_fit, '-k', 'DisplayName', '����');  % �������

% ���Ƶ��������ݼ��������
%plot(x, y2, 'or', 'DisplayName', '����������');  % ԭʼ����ɢ��ͼ
plot(x, y2_fit, '-r', 'DisplayName', 'x����');  % �������

% ���Ƶ��������ݼ��������
%plot(x, y3, 'og', 'DisplayName', '����������');  % ԭʼ����ɢ��ͼ
plot(x, y3_fit, '-g', 'DisplayName', 'y����');  % �������

% ��ӱ�ǩ��ͼ��
xlabel('k4');
ylabel('���/m');
legend('show');  % ��ʾͼ��
%grid on;  % ��ʾ����
