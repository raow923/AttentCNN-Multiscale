#ifndef DATA_H
#define DATA_H

#include <QObject>

#define Head0	0xaa
#define Head1	0xbb
#define Tail0	0xcc
#define Tail1	0xdd

class Data : public QObject
{
    Q_OBJECT
public:
    explicit Data(QObject *parent = nullptr);

    QByteArray packRXZ(int16_t R, int16_t X);           //打包数据 添加帧头
    void short2byte(short n, QByteArray *ba, int m);    //将一个16位数据放到数组里面

signals:

};

#endif // DATA_H
