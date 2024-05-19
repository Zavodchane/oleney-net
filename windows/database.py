from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.orm import sessionmaker, Mapped, mapped_column

from windows.metadata import get_image_metadata

engine = create_engine("sqlite:///images.db")


class Base(DeclarativeBase):
    pass


class Image(Base):
    __tablename__ = "image"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)

    name: Mapped[str] = mapped_column(nullable=False)

    make: Mapped[str] = mapped_column(nullable=True)
    model: Mapped[str] = mapped_column(nullable=True)

    datetime: Mapped[str] = mapped_column(nullable=True)

    class_: Mapped[str] = mapped_column(nullable=True)

    hash: Mapped[str] = mapped_column(nullable=True)


# Base.metadata.drop_all(engine)
Base.metadata.create_all(engine)

Session = sessionmaker(bind=engine)

session = Session()


def save_imgs_metadata(imgs_path: list, classes: list):
    for i in range(len(imgs_path)):
        img_metadata = get_image_metadata(imgs_path[i])
        img_metadata["class_"] = classes[i]
        session.add(Image(**img_metadata))
    session.commit()
