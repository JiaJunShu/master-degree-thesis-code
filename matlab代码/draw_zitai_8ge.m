% �����ļ���
filename = 'output.txt'; % �����ʵ���ļ����޸�

% ʹ�� readtable ��ȡ�ļ�
dataTable = readtable(filename);

% ��ȡ��������
seventhColumn = dataTable{:, 7}; % ������
firstColumn = dataTable{:, 1}; % ��һ��
secondColumn = dataTable{:, 2}; % �ڶ���
thirdColumn = dataTable{:, 3}; % ������
eighthColumn = dataTable{:, 8}; % �ڰ���
ninthColumn = dataTable{:, 9}; % �ھ���
tenthColumn = dataTable{:, 10}; % ��ʮ��
eleventhColumn = dataTable{:, 11}; % ��ʮһ��
twelfthColumn = dataTable{:, 12}; % ��ʮ����

% ��鲢ת����Ϊ��ֵ��ʽ
columnsToCheck = {seventhColumn, firstColumn, secondColumn, thirdColumn, eighthColumn, ninthColumn, tenthColumn, eleventhColumn, twelfthColumn};
for i = 1:length(columnsToCheck)
    if iscell(columnsToCheck{i})
        columnsToCheck{i} = str2double(columnsToCheck{i});
    else
        columnsToCheck{i} = double(columnsToCheck{i});
    end
end

% ������ת������
[seventhColumn, firstColumn, secondColumn, thirdColumn, eighthColumn, ninthColumn, tenthColumn, eleventhColumn, twelfthColumn] = deal(columnsToCheck{:});

% ɾ���κ� NaN ֵ��ȷ�������Ǹɾ���
validIndices = ~any(isnan([seventhColumn, firstColumn, secondColumn, thirdColumn, eighthColumn, ninthColumn, tenthColumn, eleventhColumn, twelfthColumn]), 2);
seventhColumn = seventhColumn(validIndices);
firstColumn = firstColumn(validIndices);
secondColumn = secondColumn(validIndices);
thirdColumn = thirdColumn(validIndices);
eighthColumn = eighthColumn(validIndices);
ninthColumn = ninthColumn(validIndices);
tenthColumn = tenthColumn(validIndices);
eleventhColumn = eleventhColumn(validIndices);
twelfthColumn = twelfthColumn(validIndices);

% ����ͼ��
figure; % ����һ����ͼ�δ���
set(gcf, 'Color', 'w'); % ��������ͼ�δ��ڱ���Ϊ��ɫ
plot(seventhColumn, firstColumn, 'k-', 'DisplayName', '��һ��'); % ��һ������
xlabel('t(s)'); % x ���ǩ
ylabel('��Ͳ��ת��/��'); % y ���ǩ
%grid on; % �������

figure; % ������һ��ͼ�δ���
set(gcf, 'Color', 'w'); % ���ñ���Ϊ��ɫ
plot(seventhColumn, secondColumn, 'k-', 'DisplayName', '�ڶ���'); % �ڶ�������
xlabel('t(s)'); % x ���ǩ
ylabel('�����ת��/��'); % y ���ǩ
%grid on; % �������

figure; % ������һ��ͼ�δ���
set(gcf, 'Color', 'w'); % ���ñ���Ϊ��ɫ
plot(seventhColumn, thirdColumn, 'k-', 'DisplayName', '������'); % ����������
xlabel('t(s)'); % x ���ǩ
ylabel('������/��'); % y ���ǩ
%grid on; % �������

% ���Ƶڰ���ͼ��
figure; % ����һ����ͼ�δ���
set(gcf, 'Color', 'w'); % ��������ͼ�δ��ڱ���Ϊ��ɫ
plot(seventhColumn, eighthColumn, 'k-', 'DisplayName', '�ڰ���'); % �ڰ�������
xlabel('t(s)'); % x ���ǩ
ylabel('ƫ����/��'); % y ���ǩ
grid off; % �ر�����

% ���Ƶھ���ͼ��
figure; % ������һ����ͼ�δ���
set(gcf, 'Color', 'w'); % ���ñ���Ϊ��ɫ
plot(seventhColumn, ninthColumn, 'k-', 'DisplayName', '�ھ���'); % �ھ�������
xlabel('t(s)'); % x ���ǩ
ylabel('�ھ�������'); % y ���ǩ
grid off; % �ر�����

% ���Ƶ�ʮ��ͼ��
figure; % ����һ����ͼ�δ���
set(gcf, 'Color', 'w'); % ���ñ���Ϊ��ɫ
plot(seventhColumn, tenthColumn, 'k-', 'DisplayName', '��ʮ��'); % ��ʮ������
xlabel('t(s)'); % x ���ǩ
ylabel('��ʮ������'); % y ���ǩ
grid off; % �ر�����

% ���Ƶ�ʮһ��ͼ��
figure; % ����һ����ͼ�δ���
set(gcf, 'Color', 'w'); % ���ñ���Ϊ��ɫ
plot(seventhColumn, eleventhColumn, 'k-', 'DisplayName', '��ʮһ��'); % ��ʮһ������
xlabel('t(s)'); % x ���ǩ
ylabel('�������ٶ�(��/s)'); % y ���ǩ
grid off; % �ر�����

% ���Ƶ�ʮ����ͼ��
figure; % ����һ����ͼ�δ���
set(gcf, 'Color', 'w'); % ���ñ���Ϊ��ɫ
plot(seventhColumn, twelfthColumn, 'k-', 'DisplayName', '��ʮ����'); % ��ʮ��������
xlabel('t(s)'); % x ���ǩ
ylabel('ƫ�����ٶ�(��/s)'); % y ���ǩ
grid off; % �ر�����
