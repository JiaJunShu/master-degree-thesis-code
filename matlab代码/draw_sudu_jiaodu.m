% �����ļ���
filename = 'output.txt'; % �����ʵ���ļ����޸�

% ����1: ʹ�� readtable ��ȡ�ļ�
dataTable = readtable(filename);

% ��ȡ����������
seventhColumn = dataTable{:, 7}; % ������
fourteenthColumn = dataTable{:, 14}; % ��ʮ����
thirteenthColumn = dataTable{:, 13}; % ��ʮ����

% ��鲢ת��������Ϊ��Ԫ�����
if iscell(seventhColumn)
    seventhColumn = str2double(seventhColumn);
else
    seventhColumn = double(seventhColumn);
end

if iscell(fourteenthColumn)
    fourteenthColumn = str2double(fourteenthColumn);
else
    fourteenthColumn = double(fourteenthColumn);
end

if iscell(thirteenthColumn)
    thirteenthColumn = str2double(thirteenthColumn);
else
    thirteenthColumn = double(thirteenthColumn);
end

% ɾ���κ� NaN ֵ��ȷ�������Ǹɾ���
validIndices = ~(isnan(seventhColumn) | isnan(fourteenthColumn) | isnan(thirteenthColumn));
seventhColumn = seventhColumn(validIndices);
fourteenthColumn = fourteenthColumn(validIndices);
thirteenthColumn = thirteenthColumn(validIndices);

% ���Ƶ���������ͼ
figure; % ����һ����ͼ�δ���
set(gcf, 'Color', 'w'); % ��������ͼ�δ��ڱ���Ϊ��ɫ
plot(seventhColumn, fourteenthColumn, 'k-', 'DisplayName', '��ʮ��������'); % ʹ�ú�ɫʵ�߻���
xlabel('t(s)'); % Ϊ x ����ӱ�ǩ
ylabel('������б��(��)'); % Ϊ y ����ӱ�ǩ
%title('������Ϊ������ ��ʮ��������ͼ'); % ��ӱ���
%grid on; % �������
%legend('show'); % ����ѡ����ʾͼ��

% ���Ƶ�ʮ��������ͼ
figure; % ������һ����ͼ�δ���
set(gcf, 'Color', 'w'); % ��������ͼ�δ��ڱ���Ϊ��ɫ
plot(seventhColumn, thirteenthColumn, 'k-', 'DisplayName', '��ʮ��������'); % ʹ�ú�ɫʵ�߻���
xlabel('t(s)'); % Ϊ x ����ӱ�ǩ
ylabel('������λ��(��)'); % Ϊ y ����ӱ�ǩ
%title('������Ϊ������ ��ʮ��������ͼ'); % ��ӱ���
%grid on; % �������
%legend('show'); % ����ѡ����ʾͼ��
