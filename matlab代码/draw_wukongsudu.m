% �����ļ���
filename = 'output.txt'; % �����ʵ���ļ����޸�

% ����1: ʹ�� readtable ��ȡ�ļ�
dataTable = readtable(filename);

% ��ȡ����������
seventhColumn = dataTable{:, 7}; % ������
fifteenthColumn = dataTable{:, 15}; % ��ʮ���У�ȷ���ļ�������һ�У�

% ��鲢ת��������Ϊ��Ԫ�����
if iscell(seventhColumn)
    seventhColumn = str2double(seventhColumn);
else
    seventhColumn = double(seventhColumn);
end

if iscell(fifteenthColumn)
    fifteenthColumn = str2double(fifteenthColumn);
else
    fifteenthColumn = double(fifteenthColumn);
end

% ɾ���κ� NaN ֵ��ȷ�������Ǹɾ���
validIndices = ~(isnan(seventhColumn) | isnan(fifteenthColumn));
seventhColumn = seventhColumn(validIndices);
fifteenthColumn = fifteenthColumn(validIndices);

% ����ͼ��
figure; % ����һ����ͼ�δ���
set(gcf, 'Color', 'w'); % ��������ͼ�δ��ڱ���Ϊ��ɫ
plot(seventhColumn, fifteenthColumn, 'k-', 'DisplayName', '��ʮ��������'); % ʹ�ú�ɫʵ�߻���
xlabel('t(s)'); % Ϊ x ����ӱ�ǩ
ylabel('v(m/s)'); % Ϊ y ����ӱ�ǩ
%title('������Ϊ������ ��ʮ��������ͼ'); % ��ӱ���
%grid on; % �������
%legend('show'); % ��ʾͼ��
