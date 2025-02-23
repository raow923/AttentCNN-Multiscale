#include "widget.h"

#include <QApplication>
#include <QSharedMemory>
#include <QMessageBox>

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);

    QSharedMemory shared_memory;
    shared_memory.setKey(QString("SRH_V1"));
    if(shared_memory.attach())//共享内存被占用则直接返回
    {
        QMessageBox::information(NULL,QStringLiteral("提示"),QStringLiteral("已经打开一个本应用！"));
        return 0;
    }
    shared_memory.create(1);//共享内存没有被占用则创建UI
    Widget w;
    w.show();
    w.setWindowTitle("SRH_V1");
    return a.exec();
}
