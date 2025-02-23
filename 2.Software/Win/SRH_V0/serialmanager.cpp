#include "serialmanager.h"
#include "ui_serialmanager.h"

#include "comchange.h"
#include "QDebug"

serialManager::serialManager(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::serialManager)
{
    ui->setupUi(this);

    void(QSpinBox:: * spSignal )(int) = &QSpinBox::valueChanged;
    connect(ui->Hand, &QSlider::valueChanged,ui->Value,&QSpinBox::setValue);//QSlider滑动  QSpinBox数字跟着改变
    connect(ui->Value , spSignal , ui->Hand , &QSlider::setValue);
    connect(ui->Hand, &QSlider::sliderReleased,this,&serialManager::sendData);

    //串口插拔检测
    ComChange::getInstance()->setHWND((HWND)this->winId());
    connect(ComChange::getInstance(), &ComChange::comStatus, this, &serialManager::on_comStatus);
    setLED(ui->LED, 1, 30); //设置串口灯
    /* 新建串口类 */
    //串口设置
    m_SerialPort = new QSerialPort();
    m_SerialPort->setBaudRate(115200);
    m_SerialPort->setDataBits(QSerialPort::Data8);
    m_SerialPort->setParity(QSerialPort::NoParity);
    m_SerialPort->setStopBits(QSerialPort::OneStop);
    m_SerialPort->setFlowControl(QSerialPort::NoFlowControl);

    //将可用串口显示在控件上
    on_ButtonScan_clicked();
}

serialManager::~serialManager()
{
    delete ui;
}

/************************************************
 * 函数名：    setLED
 * 函数功能：  将label控件设置为LED灯
 * 输入参数：  *label 待设置的label, color颜色0灰色1红色2绿色3黄色,  size控件大小
 * 输出参数：  void
 * 返回值：    void
************************************************/
void serialManager::setLED(QLabel *label, int color, int size)
{
    // 将label中的文字清空
    label->setText("");
    // 先设置矩形大小
    // 如果ui界面设置的label大小比最小宽度和高度小，矩形将被设置为最小宽度和最小高度；
    // 如果ui界面设置的label大小比最小宽度和高度大，矩形将被设置为最大宽度和最大高度；
    QString min_width = QString("min-width: %1px;").arg(size);              // 最小宽度：size
    QString min_height = QString("min-height: %1px;").arg(size);            // 最小高度：size
    QString max_width = QString("max-width: %1px;").arg(size);              // 最小宽度：size
    QString max_height = QString("max-height: %1px;").arg(size);            // 最小高度：size
    // 再设置边界形状及边框
    QString border_radius = QString("border-radius: %1px;").arg(size/2);    // 边框是圆角，半径为size/2
    QString border = QString("border:1px solid black;");                    // 边框为1px黑色
    // 最后设置背景颜色
    QString background = "background-color:";
    switch (color)
    {
    case 0:
        // 灰色
        background += "rgb(190,190,190)";
        break;
    case 1:
        // 红色
        background += "rgb(255,0,0)";
        break;
    case 2:
        // 绿色
        background += "rgb(0,255,0)";
        break;
    case 3:
        // 黄色
        background += "rgb(255,255,0)";
        break;
    default:
        break;
    }
    const QString SheetStyle = min_width + min_height + max_width + max_height + border_radius + border + background;
    label->setStyleSheet(SheetStyle);
}

/************************************************
 * 函数名：   readSerialPort()
 * 函数功能： 读取串口数据槽函数
 * 输入参数： void
 * 输出参数： void
 * 返回值：  void
************************************************/
void serialManager::readSerialPort()
{
    QByteArray recvData = m_SerialPort->readAll();
    if(dataType)
    {
        ui->TeditRecv->insertPlainText(QString(recvData.toHex()));
        ui->TeditRecv ->moveCursor(QTextCursor::End, QTextCursor::MoveAnchor);  //光标移动到文末
        return;
    }

    ui->TeditRecv->insertPlainText(QString(recvData));
    ui->TeditRecv ->moveCursor(QTextCursor::End, QTextCursor::MoveAnchor);  //光标移动到文末
}
/**************************************************************
 * 函数名：   on_ButtonScan_clicked
 * 函数功能： 扫描按键槽函数 扫描串口并显示
 * 输入参数： void
 * 输出参数： void
 * 返回值：   void
**************************************************************/
void serialManager::on_ButtonScan_clicked()
{
    if(m_SerialPort != NULL)
    {
        ui->Choose->clear();//将控件内容清空
        foreach (const QSerialPortInfo &info,QSerialPortInfo::availablePorts())
        {
            // 排除蓝牙端口
            if (info.description().contains("蓝牙", Qt::CaseInsensitive))
                continue;
            QSerialPort serial;
            serial.setPort(info);
            if(serial.open(QIODevice::ReadWrite))//仅显示可用串口
            {
                ui->Choose->addItem((serial.portName()));//添加串口到控件显示
                serial.close();
            }
        }
    }
}
/**************************************************************
 * 函数名：   on_ButtonOpen_clicked
 * 函数功能： 打开按键槽函数 将选中的串口打开
 * 输入参数： void
 * 输出参数： void
 * 返回值：   void
**************************************************************/
void serialManager::on_ButtonOpen_clicked()
{
    if(m_SerialPort->isOpen())//串口打开进
    {
        m_SerialPort->close();
        ui->ButtonOpen->setText("Start");
        disconnect(m_SerialPort, &QSerialPort::readyRead, this, &serialManager::readSerialPort);
        setLED(ui->LED, 1, 30);
    }
    else    //串口未打开进
    {
        com = ui->Choose->currentText();
        m_SerialPort->setPortName(com);
        m_SerialPort->open(QSerialPort::ReadWrite);
        if(m_SerialPort->isOpen())
        {
            ui->ButtonOpen->setText("Stop");
            connect(m_SerialPort, &QSerialPort::readyRead, this, &serialManager::readSerialPort);
        }
    }
}

void serialManager::on_ButtonClawClose_clicked()
{
    if(m_SerialPort->isOpen())//串口打开进
    {
        m_SerialPort->write(m_data->packRXZ(0x0000,00));
        ui->Hand->setValue(0x0);
    }
}

void serialManager::on_ButtonClawOpen_clicked()
{
    if(m_SerialPort->isOpen())//串口打开进
    {
        m_SerialPort->write(m_data->packRXZ(0x00a0,00));
        ui->Hand->setValue(0xa0);
    }
}

void serialManager::on_ButtonStop_clicked()
{
    if(m_SerialPort->isOpen())//串口打开进
    {
        qint16 HandData;
        QByteArray SendData;
        HandData = ui->Hand->value();
        SendData = m_data->packRXZ(HandData,0xaa00);
        m_SerialPort->write(SendData);
    }
}

void serialManager::on_ButtonStart_clicked()
{
    if(m_SerialPort->isOpen())//串口打开进
    {
        qint16 HandData;
        QByteArray SendData;
        HandData = ui->Hand->value();
        SendData = m_data->packRXZ(HandData,0xbb00);
        m_SerialPort->write(SendData);
    }
}

void serialManager::on_Mode1_clicked()
{
    if(m_SerialPort->isOpen())//串口打开进
    {
        qint16 HandData;
        QByteArray SendData;
        HandData = ui->Hand->value();
        SendData = m_data->packRXZ(HandData,0x00aa);
        m_SerialPort->write(SendData);
    }
}

void serialManager::on_Mode2_clicked()
{
    if(m_SerialPort->isOpen())//串口打开进
    {
        qint16 HandData;
        QByteArray SendData;
        HandData = ui->Hand->value();
        SendData = m_data->packRXZ(HandData,0x00bb);
        m_SerialPort->write(SendData);
    }
}

void serialManager::on_Mode3_clicked()
{
    if(m_SerialPort->isOpen())//串口打开进
    {
        qint16 HandData;
        QByteArray SendData;
        HandData = ui->Hand->value();
        SendData = m_data->packRXZ(HandData,0x00cc);
        m_SerialPort->write(SendData);
    }
}

void serialManager::on_Mode4_clicked()
{
    if(m_SerialPort->isOpen())//串口打开进
    {
        qint16 HandData;
        QByteArray SendData;
        HandData = ui->Hand->value();
        SendData = m_data->packRXZ(HandData,0x00dd);
        m_SerialPort->write(SendData);
    }
}

void serialManager::on_ButtonClean_clicked()
{
    ui->TeditRecv->clear();
}

void serialManager::on_ButtonSend_clicked()
{
    if(m_SerialPort->isOpen())//串口打开进
    {
        m_SerialPort->write(ui->TeditSend->toPlainText().toStdString().c_str());
    }
}

void serialManager::on_checkBox_stateChanged(int arg1)
{
    if(arg1)
    {
        dataType = 1;
        return;
    }
    dataType = 0;
}

/**************************************************************
 * 函数名：   on_comStatus
 * 函数功能： 串口拔插槽函数
 * 输入参数： name 操作的串口名 flag 拔插判断
 * 输出参数： void
 * 返回值：   void
**************************************************************/
void serialManager::on_comStatus(QString name, bool flag)
{
    on_ButtonScan_clicked();
    if(flag)              // 串口插入时
    {
//        on_ButtonScan_clicked();
    }
    else                  // 串口拔出时关闭串口
    {
        if(name == com)
        {
            m_SerialPort->close();
            ui->ButtonOpen->setText("Start");
            disconnect(m_SerialPort, &QSerialPort::readyRead, this, &serialManager::readSerialPort);
            setLED(ui->LED, 1, 30);
        }
    }
}

void serialManager::sendData()
{
    qint16 HandData;
    QByteArray SendData;
    HandData = ui->Hand->value();
    SendData = m_data->packRXZ(HandData,00);
    m_SerialPort->write(SendData);
}
