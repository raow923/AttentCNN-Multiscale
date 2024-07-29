#ifndef SERIALMANAGER_H
#define SERIALMANAGER_H

#include <QWidget>

#include "QLabel"
#include <QSerialPort>
#include <QSerialPortInfo>
#include "data.h"

namespace Ui {
class serialManager;
}

class serialManager : public QWidget
{
    Q_OBJECT

public:
    explicit serialManager(QWidget *parent = nullptr);
    ~serialManager();

    void setLED(QLabel* label, int color, int size);   //将label控件变成一个圆形指示灯，需要指定颜色color以及直径size
    void readSerialPort();                             //读取串口

private slots:
    void on_ButtonScan_clicked();

    void on_ButtonOpen_clicked();

    void on_ButtonClawClose_clicked();

    void on_ButtonClawOpen_clicked();

    void on_ButtonStop_clicked();

    void on_ButtonStart_clicked();

    void on_Mode1_clicked();

    void on_Mode2_clicked();

    void on_Mode3_clicked();

    void on_Mode4_clicked();

    void on_ButtonClean_clicked();

    void on_ButtonSend_clicked();

    void on_checkBox_stateChanged(int arg1);

    void on_comStatus(QString name, bool flag);

    void sendData();

private:
    Ui::serialManager *ui;

    QSerialPort *m_SerialPort;                 //串口
    QString com;                               //打开的串口号
    Data *m_data;                              //数据打包对象
    bool dataType =0;
};

#endif // SERIALMANAGER_H
